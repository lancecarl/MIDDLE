#!/usr/bin/env python3
"""Generate temporal train/validation/test splits for UPFD datasets."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import numpy as np
from collections import defaultdict


DATASET_PREFIX = {
    'gossipcop':  'gos',
    'politifact': 'pol',
}


def _load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _resolve_pub_times(
    n_graphs: int,
    node_graph_id: np.ndarray,
    time_mapping: dict,
    root_nodes: frozenset,
) -> np.ndarray:
    """Run _resolve_pub_times."""

    graph2nodes: dict[int, list[int]] = defaultdict(list)
    for node_id, g_idx in enumerate(node_graph_id):
        graph2nodes[int(g_idx)].append(node_id)

    pub_times = np.full(n_graphs, np.inf, dtype=np.float64)

    for g_idx in range(n_graphs):
        nodes = graph2nodes[g_idx]
        min_ts = np.inf
        root_ts = None

        for nid in nodes:
            ts_raw = time_mapping.get(nid, '')
            if ts_raw == '' or ts_raw is None:
                if nid in root_nodes:
                    root_ts = None
                continue
            try:
                ts = int(ts_raw)
            except (ValueError, TypeError):
                continue

            if nid in root_nodes:
                root_ts = ts
            if ts < min_ts:
                min_ts = ts

        if root_ts is not None:
            pub_times[g_idx] = root_ts
        elif min_ts < np.inf:
            pub_times[g_idx] = min_ts


    return pub_times


def generate_split(
    root: str,
    dataset: str,
    train_ratio: float = 0.7,
    val_ratio: float   = 0.1,
    dry_run: bool      = False,
    overwrite: bool    = True,
) -> dict:
    """Run generate_split."""
    assert dataset in DATASET_PREFIX, f"Unknown dataset: {dataset}"
    prefix  = DATASET_PREFIX[dataset]
    raw_dir = osp.join(root, dataset, 'raw')


    labels       = np.load(osp.join(raw_dir, 'graph_labels.npy'))
    node_graph_id = np.load(osp.join(raw_dir, 'node_graph_id.npy'))
    n_graphs     = len(labels)

    time_mapping = _load_pkl(
        osp.join(raw_dir, f'{prefix}_id_time_mapping.pkl')
    )

    time_mapping = {int(k): v for k, v in time_mapping.items()}


    root_nodes = frozenset(
        nid for nid, ts in time_mapping.items()
        if ts == '' or ts is None
    )

    print(f"Dataset: {dataset} | graphs: {n_graphs} | root nodes: {len(root_nodes)}")


    pub_times = _resolve_pub_times(
        n_graphs, node_graph_id, time_mapping, root_nodes
    )

    n_unknown = int(np.sum(np.isinf(pub_times)))
    if n_unknown:
        print(f"{n_unknown} graphs have unknown publication time and will be placed last")


    sorted_indices = np.argsort(pub_times, kind='stable')


    n_train = int(n_graphs * train_ratio)
    n_val   = int(n_graphs * val_ratio)
    n_test  = n_graphs - n_train - n_val

    train_idx = sorted_indices[:n_train].astype(np.int64)
    val_idx   = sorted_indices[n_train: n_train + n_val].astype(np.int64)
    test_idx  = sorted_indices[n_train + n_val:].astype(np.int64)

    assert len(train_idx) + len(val_idx) + len(test_idx) == n_graphs, \
        "Split sizes do not sum to the number of graphs."


    def _label_dist(idx):
        lbs = labels[idx]
        n_fake = int((lbs == 1).sum())
        n_real = int((lbs == 0).sum())
        return f"fake={n_fake}, real={n_real} (ratio {n_fake/max(len(idx),1)*100:.1f}%)"

    def _ts_range(idx):
        ts = pub_times[idx]
        valid = ts[~np.isinf(ts)]
        if len(valid) == 0:
            return "N/A"
        lo, hi = int(valid.min()), int(valid.max())

        try:
            import datetime
            lo_s = datetime.datetime.utcfromtimestamp(lo).strftime('%Y-%m-%d')
            hi_s = datetime.datetime.utcfromtimestamp(hi).strftime('%Y-%m-%d')
            return f"{lo_s} → {hi_s}"
        except Exception:
            return f"{lo} → {hi}"

    print(f"\n{'='*60}")
    print(f"  Temporal split ({train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%})")
    print(f"{'='*60}")
    print(f"  train: {len(train_idx):5d} graphs | {_label_dist(train_idx)} | {_ts_range(train_idx)}")
    print(f"  val:   {len(val_idx):5d} graphs | {_label_dist(val_idx)}   | {_ts_range(val_idx)}")
    print(f"  test:  {len(test_idx):5d} graphs | {_label_dist(test_idx)}  | {_ts_range(test_idx)}")
    print(f"{'='*60}\n")

    if dry_run:
        print("--dry_run mode: no files written")
        return {'train': train_idx, 'val': val_idx, 'test': test_idx}


    out = {
        'train': (osp.join(raw_dir, 'custom_train_idx.npy'), train_idx),
        'val':   (osp.join(raw_dir, 'custom_val_idx.npy'),   val_idx),
        'test':  (osp.join(raw_dir, 'custom_test_idx.npy'),  test_idx),
    }

    for split, (path, arr) in out.items():
        if osp.exists(path) and not overwrite:
            print(f"File exists; skipping because --overwrite is false: {path}")
            continue
        np.save(path, arr)
        print(f"Wrote: {path} ({len(arr)} graphs)")

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Generate DAWN-style 7:1:2 temporal split indices by publication time"
    )
    parser.add_argument('--root',        required=True,
                        help="Data root directory, e.g., /path/to/data")
    parser.add_argument('--dataset',     required=True,
                        choices=['gossipcop', 'politifact'])
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio',   type=float, default=0.1)
    parser.add_argument('--dry_run',     action='store_true',
                        help="Print statistics without writing files")
    parser.add_argument('--no_overwrite', action='store_true',
                        help="Skip existing files")
    args = parser.parse_args()

    generate_split(
        root=args.root,
        dataset=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
    )
