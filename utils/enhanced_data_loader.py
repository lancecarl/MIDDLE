"""Temporal UPFD data loader with user alignment fields."""

import os
import os.path as osp
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.sparse import csr_matrix
from collections import defaultdict
from tqdm import tqdm


def _load_npz_feature(path: str) -> csr_matrix:
    """Run _load_npz_feature."""
    d = np.load(path, allow_pickle=True)
    return csr_matrix(
        (d['data'], d['indices'], d['indptr']),
        shape=tuple(d['shape'])
    )


def _load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


class TemporalUPFDDataLoaderOptimized:
    """TemporalUPFDDataLoaderOptimized module."""

    DATASET_PREFIX = {
        'gossipcop': 'gos',
        'politifact': 'pol',
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature: str = 'bert',
        use_custom_split: bool = True,

        train_ratio: float = 0.7,
        val_ratio: float = 0.1,

        temporal_context_aware: bool = True,


        # cutoff[i] = pub_time[i] + dead_criterion_hours * 3600

        dead_criterion_hours: float = 12.0,
    ):
        assert name in self.DATASET_PREFIX, f"Unknown dataset: {name}"
        self.root = root
        self.name = name
        self.feature = feature
        self.use_custom_split = use_custom_split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.temporal_context_aware = temporal_context_aware
        self.dead_criterion_seconds = dead_criterion_hours * 3600.0

        self.raw_dir = osp.join(root, name, 'raw')
        self.prefix = self.DATASET_PREFIX[name]


        self.labels = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        self.node_graph_id = np.load(
            osp.join(self.raw_dir, 'node_graph_id.npy')
        )
        self.n_graphs = len(self.labels)
        self.n_nodes = len(self.node_graph_id)


        self.time_mapping = _load_pkl(
            osp.join(self.raw_dir, f'{self.prefix}_id_time_mapping.pkl')
        )

        self.time_mapping = {int(k): v for k, v in self.time_mapping.items()}


        twitter_path = osp.join(
            self.raw_dir, f'{self.prefix}_id_twitter_mapping.pkl'
        )
        if osp.exists(twitter_path):
            self.twitter_mapping = _load_pkl(twitter_path)
            self.twitter_mapping = {
                int(k): v for k, v in self.twitter_mapping.items()
            }
        else:
            print(f"Twitter mapping file not found: {twitter_path}")
            self.twitter_mapping = {}


        feat_path = osp.join(
            self.raw_dir, f'new_{feature}_feature.npz'
        )
        self.feature_matrix = _load_npz_feature(feat_path)
        self.feature_dim = self.feature_matrix.shape[1]
        print(f"Loaded features [{feature}]: shape={self.feature_matrix.shape}")


        self.edges_global = self._load_edges()


        self._graph_node_ranges = self._precompute_node_ranges()


        self._root_nodes = frozenset(
            nid for nid, ts in self.time_mapping.items()
            if ts == '' or ts is None
        )


        self._graph_cutoffs = self._precompute_graph_cutoffs()


        self.split_indices = self._load_split_indices()


        self._graphs = None

        print(
            f"Dataset [{name}] loaded: "
            f"{self.n_graphs} graphs, {self.n_nodes} nodes, "
            f"{len(self.twitter_mapping)} cross-graph user mappings | "
            f"temporal filter: {'enabled' if temporal_context_aware else 'disabled'}"
            + (f" (dead_criterion={dead_criterion_hours}h)" if temporal_context_aware else "")
        )


    def _load_edges(self) -> np.ndarray:
        """Run _load_edges."""
        edges = []
        with open(osp.join(self.raw_dir, 'A.txt')) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        edges.append([int(parts[0].strip()), int(parts[1].strip())])
                    except ValueError:
                        pass
        return np.array(edges, dtype=np.int64)

    def _precompute_node_ranges(self) -> list:
        """Run _precompute_node_ranges."""
        ranges = []
        for graph_idx in range(self.n_graphs):
            node_ids = np.where(self.node_graph_id == graph_idx)[0]
            ranges.append(node_ids)
        return ranges


    def _precompute_graph_cutoffs(self) -> np.ndarray:
        """Run _precompute_graph_cutoffs."""
        cutoffs = np.full(self.n_graphs, np.inf, dtype=np.float64)

        for g_idx, node_ids in enumerate(self._graph_node_ranges):
            root_ts   = None
            min_user_ts = np.inf

            for nid in node_ids:
                nid = int(nid)
                ts_raw = self.time_mapping.get(nid, '')
                if ts_raw == '' or ts_raw is None:


                    continue
                try:
                    ts = int(ts_raw)
                except (ValueError, TypeError):
                    continue

                if nid in self._root_nodes:

                    root_ts = ts
                else:

                    if ts < min_user_ts:
                        min_user_ts = ts


            if root_ts is not None:
                pub_time = float(root_ts)
            elif min_user_ts < np.inf:
                pub_time = float(min_user_ts)
            else:

                continue

            cutoffs[g_idx] = pub_time + self.dead_criterion_seconds

        return cutoffs


    def _load_split_indices(self) -> dict:
        if self.use_custom_split:
            custom_train = osp.join(self.raw_dir, 'custom_train_idx.npy')
            custom_val   = osp.join(self.raw_dir, 'custom_val_idx.npy')
            custom_test  = osp.join(self.raw_dir, 'custom_test_idx.npy')
            if osp.exists(custom_train):
                indices = {
                    'train': np.load(custom_train),
                    'val':   np.load(custom_val),
                    'test':  np.load(custom_test),
                }
                print(
                    f"Loaded custom split: "
                    f"train={len(indices['train'])}, "
                    f"val={len(indices['val'])}, "
                    f"test={len(indices['test'])}"
                )
                return indices
            else:
                print(
                    f"custom_train_idx.npy not found; using fallback temporal split "
                    f"({self.train_ratio:.0%}/{self.val_ratio:.0%}/"
                    f"{1-self.train_ratio-self.val_ratio:.0%})。"
                    f"\n   Suggested command: python generate_temporal_split.py "
                    f"--root {self.root} --dataset {self.name}"
                )


        n = self.n_graphs
        idx = np.arange(n)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)
        return {
            'train': idx[:n_train],
            'val':   idx[n_train: n_train + n_val],
            'test':  idx[n_train + n_val:],
        }


    def _build_graph(
        self,
        graph_idx: int,
        temporal_cutoff: float = None,
    ) -> Data:
        """Run _build_graph."""
        global_nodes = self._graph_node_ranges[graph_idx]  # ndarray [N_full]


        if self.temporal_context_aware and temporal_cutoff is not None:
            keep_mask = self._compute_temporal_mask(global_nodes, temporal_cutoff)
            global_nodes = global_nodes[keep_mask]
        else:
            keep_mask = None

        n_local = len(global_nodes)


        id_map = {int(g): local for local, g in enumerate(global_nodes)}
        node_set = set(global_nodes.tolist())


        src_g = self.edges_global[:, 0]
        dst_g = self.edges_global[:, 1]
        mask = np.array([s in node_set and d in node_set
                         for s, d in zip(src_g, dst_g)], dtype=bool)
        if mask.any():
            local_src = np.array([id_map[int(s)] for s in src_g[mask]])
            local_dst = np.array([id_map[int(d)] for d in dst_g[mask]])
            edge_index = torch.tensor(
                np.stack([local_src, local_dst], axis=0), dtype=torch.long
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)


        feat = self.feature_matrix[global_nodes].toarray()  # (N_local, D)
        x = torch.tensor(feat, dtype=torch.float32)


        y = torch.tensor(int(self.labels[graph_idx]), dtype=torch.long)


        time_map_local = {}
        for local_idx, global_nid in enumerate(global_nodes):
            ts = self.time_mapping.get(int(global_nid), '')
            time_map_local[local_idx] = ts


        twitter_ids = []
        for global_nid in global_nodes:
            tw_str = self.twitter_mapping.get(int(global_nid), None)
            if tw_str is None:
                twitter_ids.append(-1)
            else:
                try:
                    twitter_ids.append(int(tw_str))
                except (ValueError, TypeError):
                    twitter_ids.append(-1)


        global_node_ids_tensor = torch.tensor(
            global_nodes.astype(np.int64), dtype=torch.long
        )
        twitter_ids_tensor = torch.tensor(twitter_ids, dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            time_mapping=time_map_local,
            twitter_ids=twitter_ids_tensor,
            global_node_ids=global_node_ids_tensor,
            graph_idx=torch.tensor(graph_idx, dtype=torch.long),
        )
        return data

    def _compute_temporal_mask(
        self,
        global_nodes: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """Run _compute_temporal_mask."""
        mask = np.ones(len(global_nodes), dtype=bool)

        for i, nid in enumerate(global_nodes):
            nid = int(nid)


            if nid in self._root_nodes:
                continue

            ts_raw = self.time_mapping.get(nid, '')
            if ts_raw == '' or ts_raw is None:

                continue

            try:
                ts = int(ts_raw)
            except (ValueError, TypeError):

                continue

            if ts > cutoff:
                mask[i] = False

        return mask


    def _ensure_graphs_built(self):
        if self._graphs is None:
            dc_h = self.dead_criterion_seconds / 3600.0
            print(
                f"Building {self.n_graphs} graphs "
                f"(temporal_context_aware={self.temporal_context_aware}"
                + (f", dead_criterion={dc_h:.0f}h)" if self.temporal_context_aware else ")")
            )
            graphs = []
            for i in tqdm(range(self.n_graphs), desc='building graphs'):


                cutoff = self._graph_cutoffs[i] if self.temporal_context_aware else None
                graphs.append(self._build_graph(i, temporal_cutoff=cutoff))
            self._graphs = graphs


    def get_split_graphs(self, split: str) -> list:
        """Run get_split_graphs."""
        self._ensure_graphs_built()
        indices = self.split_indices[split]
        return [self._graphs[i] for i in indices]

    def get_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        """Run get_loaders."""
        from torch.utils.data import DataLoader

        self._ensure_graphs_built()

        def make_loader(split, shuffle):
            graphs = self.get_split_graphs(split)
            return DataLoader(
                _TemporalGraphDataset(graphs),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=persistent_workers and num_workers > 0,
                collate_fn=_temporal_collate_fn,
            )

        return (
            make_loader('train', shuffle=True),
            make_loader('val',   shuffle=False),
            make_loader('test',  shuffle=False),
        )


    def get_global_user_index(self, train_only: bool = True) -> dict:
        """Run get_global_user_index."""
        self._ensure_graphs_built()


        if train_only:
            allowed_graphs = set(self.split_indices['train'].tolist())
        else:
            allowed_graphs = set(range(self.n_graphs))

        index = defaultdict(list)
        for graph_idx, data in enumerate(self._graphs):


            if graph_idx not in allowed_graphs:
                continue

            n_local = data.x.shape[0]
            for local_idx in range(n_local):
                tw_id = data.twitter_ids[local_idx].item()
                if tw_id == -1:
                    continue

                global_nid = data.global_node_ids[local_idx].item()
                ts_raw = data.time_mapping.get(local_idx, '')
                if ts_raw == '' or ts_raw is None:
                    continue
                else:
                    try:
                        ts = int(ts_raw)
                    except (ValueError, TypeError):
                        ts = None

                index[tw_id].append({
                    'graph_idx':      graph_idx,
                    'local_idx':      local_idx,
                    'global_node_id': global_nid,
                    'timestamp':      ts,
                })

        return dict(index)


    def get_twitter_aligner_inputs(self, train_only: bool = True) -> dict:
        """Run get_twitter_aligner_inputs."""
        return {
            'user_index':    self.get_global_user_index(train_only=train_only),
            'graph_labels':  self.labels,
            'train_indices': self.split_indices['train'],
            'n_graphs':      self.n_graphs,
            'feature_dim':   self.feature_dim,
        }


    def stats(self) -> dict:
        """Run stats."""
        self._ensure_graphs_built()
        train_only_index = self.get_global_user_index(train_only=True)
        all_index        = self.get_global_user_index(train_only=False)
        cross_graph_users_train = sum(
            1 for apps in train_only_index.values()
            if len(set(a['graph_idx'] for a in apps)) > 1
        )
        return {
            'n_graphs':                    self.n_graphs,
            'n_nodes':                     self.n_nodes,
            'feature_dim':                 self.feature_dim,
            'n_train':                     len(self.split_indices['train']),
            'n_val':                       len(self.split_indices['val']),
            'n_test':                      len(self.split_indices['test']),
            'cross_graph_users_train_only': cross_graph_users_train,
            'total_unique_users_all':      len(all_index),
            'twitter_mapping_size':        len(self.twitter_mapping),
            'temporal_context_aware':      self.temporal_context_aware,
            'dead_criterion_hours':        self.dead_criterion_seconds / 3600.0,
        }


class _TemporalGraphDataset(torch.utils.data.Dataset):
    """_TemporalGraphDataset module."""
    def __init__(self, graphs: list):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def _temporal_collate_fn(batch):
    """Run _temporal_collate_fn."""
    from torch_geometric.data import Batch

    time_mappings = [data.time_mapping for data in batch]
    graph_indices = [data.graph_idx.item() for data in batch]

    for data in batch:
        del data.time_mapping

    batched = Batch.from_data_list(batch)
    batched.time_mappings = time_mappings
    batched.graph_indices = graph_indices

    for data, tm in zip(batch, time_mappings):
        data.time_mapping = tm

    return batched, time_mappings
