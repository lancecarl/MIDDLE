"""Cross-graph Twitter user alignment utilities."""

import numpy as np
import torch
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import scipy.sparse as sp
from tqdm import tqdm


# TwitterIDAligner

class TwitterIDAligner:
    """TwitterIDAligner module."""

    def __init__(
        self,
        user_index: dict,
        graph_labels: np.ndarray,
        train_indices: np.ndarray,
        n_graphs: int,
        feature_dim: int,
    ):
        self.user_index   = user_index
        self.graph_labels = graph_labels
        self.train_set    = set(train_indices.tolist())
        self.n_train      = len(train_indices)
        self.n_graphs     = n_graphs
        self.feature_dim  = feature_dim


        self.user_list: List[int] = sorted(user_index.keys())
        self.user2idx: Dict[int, int] = {
            uid: i for i, uid in enumerate(self.user_list)
        }
        self.n_users = len(self.user_list)


        self._B: Optional[dict] = None
        self._idf: Optional[np.ndarray] = None


    def build_participation_matrix(self) -> dict:
        """Run build_participation_matrix."""
        B = defaultdict(dict)

        n_filtered_out = 0
        for tw_id, appearances in self.user_index.items():
            per_graph: Dict[int, int] = {}
            for app in appearances:
                g_idx = app['graph_idx']


                if g_idx not in self.train_set:
                    n_filtered_out += 1
                    continue

                ts = app['timestamp']
                if ts is None:
                    ts = -1


                if g_idx not in per_graph or ts < per_graph[g_idx]:
                    per_graph[g_idx] = ts

            if per_graph:
                B[tw_id] = per_graph

        if n_filtered_out > 0:
            print(
                f"build_participation_matrix filtered {n_filtered_out} non-training records "
                f"(second safety check triggered; verify user_index was created with train_only=True)"
            )

        self._B = dict(B)
        print(
            f"Participation matrix built: {len(self._B)} training users, "
            f"covering {self.n_train} training graphs"
        )
        return self._B

    @property
    def B(self) -> dict:
        if self._B is None:
            self.build_participation_matrix()
        return self._B


    def compute_idf(self) -> np.ndarray:
        """Run compute_idf."""

        df = np.zeros(self.n_graphs, dtype=np.float32)
        for tw_id, per_graph in self.B.items():
            for g_idx in per_graph:

                df[g_idx] += 1.0


        N = self.n_train
        idf = np.log((N + 1) / (1 + df + 1e-8))


        self._idf = idf
        return idf

    @property
    def idf(self) -> np.ndarray:
        if self._idf is None:
            self.compute_idf()
        return self._idf


    def compute_sync_weight(
        self,
        uid_u: int,
        uid_v: int,
        graph_idx: int,
        eps: float = 3600.0,
    ) -> float:
        """Run compute_sync_weight."""
        ts_u = self.B.get(uid_u, {}).get(graph_idx, None)
        ts_v = self.B.get(uid_v, {}).get(graph_idx, None)
        if ts_u is None or ts_v is None or ts_u < 0 or ts_v < 0:
            return 0.0
        delta = abs(ts_u - ts_v)
        return math.exp(-delta / eps)


    def build_user_cooccurrence_graph(
        self,
        p_hat: Dict[int, float],
        min_cooccurrence: int = 2,
        eps: float = 3600.0,
        use_idf: bool = True,
        use_sync: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run build_user_cooccurrence_graph."""
        print(f"Building user collusion graph (users={self.n_users}, min_cooccurrence={min_cooccurrence})...")

        idf_arr = self.idf if use_idf else np.ones(self.n_graphs)


        event2users: Dict[int, List[int]] = defaultdict(list)
        predicted_events = set(p_hat.keys())
        for tw_id, per_graph in self.B.items():
            for g_idx in per_graph:
                if g_idx in predicted_events:
                    event2users[g_idx].append(tw_id)


        pair_events: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        for g_idx, users in tqdm(event2users.items(), desc='Enumerating co-participating user pairs'):
            if len(users) < 2:
                continue
            MAX_USERS_PER_EVENT = 500
            if len(users) > MAX_USERS_PER_EVENT:
                users = users[:MAX_USERS_PER_EVENT]
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u, v = users[i], users[j]
                    if u > v:
                        u, v = v, u
                    pair_events[(u, v)].append(g_idx)


        rows, cols, weights = [], [], []

        for (u, v), common_events in pair_events.items():
            if len(common_events) < min_cooccurrence:
                continue

            w = 0.0
            for g_idx in common_events:
                p_i    = p_hat.get(g_idx, 0.5)
                idf_i  = float(idf_arr[g_idx])
                sync_i = (self.compute_sync_weight(u, v, g_idx, eps)
                          if use_sync else 1.0)
                w += p_i * idf_i * sync_i

            if w <= 0:
                continue

            uid_u = self.user2idx.get(u, -1)
            uid_v = self.user2idx.get(v, -1)
            if uid_u == -1 or uid_v == -1:
                continue

            rows.append(uid_u);  cols.append(uid_v);  weights.append(w)
            rows.append(uid_v);  cols.append(uid_u);  weights.append(w)

        if not rows:
            print("No qualifying user pairs found; returning an empty graph")
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
            )

        edge_index  = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        edge_index, edge_weight = _coalesce_edges(edge_index, edge_weight, self.n_users)

        print(f"User collusion graph: {edge_index.shape[1]//2} undirected edges")
        return edge_index, edge_weight


    def aggregate_user_features(
        self,
        graphs: list,
        strategy: str = 'mean',
    ) -> torch.Tensor:
        """Run aggregate_user_features."""
        print(f"Aggregating user features (strategy={strategy}, users={self.n_users})...")

        user_feat_acc: Dict[int, List[torch.Tensor]] = defaultdict(list)
        user_ts_acc:   Dict[int, List[int]] = defaultdict(list)

        for data in graphs:
            if not hasattr(data, 'twitter_ids'):
                continue
            n_local = data.x.shape[0]
            for local_idx in range(n_local):
                tw_id = data.twitter_ids[local_idx].item()
                if tw_id == -1:
                    continue

                feat = data.x[local_idx]
                user_feat_acc[tw_id].append(feat)

                if strategy == 'latest':
                    ts_raw = data.time_mapping.get(local_idx, '')
                    try:
                        ts = int(ts_raw) if ts_raw != '' else 0
                    except (ValueError, TypeError):
                        ts = 0
                    user_ts_acc[tw_id].append(ts)

        result = torch.zeros(self.n_users, self.feature_dim, dtype=torch.float32)

        for i, tw_id in enumerate(self.user_list):
            feats = user_feat_acc.get(tw_id, [])
            if not feats:
                continue

            if strategy == 'mean':
                result[i] = torch.stack(feats).mean(dim=0)
            elif strategy == 'latest':
                ts_list  = user_ts_acc.get(tw_id, [0] * len(feats))
                best_idx = int(np.argmax(ts_list))
                result[i] = feats[best_idx]

        return result


    def filter_suspect_users(
        self,
        p_hat: Dict[int, float],
        tau_ratio: float = 0.7,
        delta: int = 3,
        phi75_percentile: float = 75.0,
        use_predictions: bool = True,
    ) -> List[int]:
        """Run filter_suspect_users."""
        scores = {}

        for tw_id, per_graph in self.B.items():
            total_w  = 0.0
            fake_w   = 0.0

            for g_idx in per_graph:

                if g_idx not in self.train_set:
                    continue

                if use_predictions:
                    p_i = p_hat.get(g_idx, 0.5)
                else:
                    p_i = float(self.graph_labels[g_idx])

                fake_w  += p_i
                total_w += 1.0

            if total_w == 0:
                continue

            R_fake = fake_w / total_w
            C_fake = fake_w

            if R_fake >= tau_ratio and C_fake >= delta:
                scores[tw_id] = R_fake * C_fake

        if not scores:
            return []

        threshold   = float(np.percentile(list(scores.values()), phi75_percentile))
        suspect_ids = [uid for uid, s in scores.items() if s >= threshold]

        print(
            f"Suspect user filtering: {len(scores)} candidates -> {len(suspect_ids)} high-risk "
            f"(tau={tau_ratio}, delta={delta}, threshold={threshold:.3f})"
        )
        return suspect_ids


    def compute_group_sync(
        self,
        group_ids: List[int],
        epsilon: float = 3600.0,
    ) -> float:
        """Run compute_group_sync."""
        if len(group_ids) < 2:
            return 0.0

        graphs_per_member = [
            set(self.B.get(uid, {}).keys())
            for uid in group_ids
        ]

        common_events = graphs_per_member[0]
        for gs in graphs_per_member[1:]:
            common_events = common_events & gs

        if not common_events:
            return 0.0

        sync_count = 0
        for g_idx in common_events:
            ts_list = []
            for uid in group_ids:
                ts = self.B.get(uid, {}).get(g_idx, None)
                if ts is not None and ts > 0:
                    ts_list.append(ts)
            if len(ts_list) >= 2:
                if max(ts_list) - min(ts_list) < epsilon:
                    sync_count += 1

        return sync_count / len(common_events)


    def stats(self) -> dict:
        cross_graph_count = sum(
            1 for tw_id, per_graph in self.B.items()
            if len(set(per_graph.keys())) > 1
        )
        avg_appearances = (
            np.mean([len(v) for v in self.B.values()])
            if self.B else 0.0
        )
        return {
            'total_users':        self.n_users,
            'cross_graph_users':  cross_graph_count,
            'avg_appearances':    round(float(avg_appearances), 2),
            'n_graphs_total':     self.n_graphs,
            'n_train_graphs':     self.n_train,
        }


def _coalesce_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run _coalesce_edges."""
    if edge_index.shape[1] == 0:
        return edge_index, edge_weight

    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    vals = edge_weight.numpy()

    pair2weight: Dict[Tuple[int, int], float] = {}
    for r, c, w in zip(rows, cols, vals):
        key = (int(r), int(c))
        if key not in pair2weight or w > pair2weight[key]:
            pair2weight[key] = float(w)

    new_rows, new_cols, new_vals = [], [], []
    for (r, c), w in pair2weight.items():
        new_rows.append(r)
        new_cols.append(c)
        new_vals.append(w)

    edge_index_out  = torch.tensor([new_rows, new_cols], dtype=torch.long)
    edge_weight_out = torch.tensor(new_vals, dtype=torch.float32)
    return edge_index_out, edge_weight_out


def build_aligner_from_loader(loader, train_only: bool = True) -> "TwitterIDAligner":
    """Run build_aligner_from_loader."""
    inputs = loader.get_twitter_aligner_inputs(train_only=train_only)
    return TwitterIDAligner(**inputs)
