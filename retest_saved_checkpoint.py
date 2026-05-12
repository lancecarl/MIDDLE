#!/usr/bin/env python3
"""Retest a saved MIRAGE Phase 3 checkpoint on the test split."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_reproducibility(seed: int = 42) -> None:
    """Set random seeds and deterministic cuDNN flags for retesting."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clean_state_dict(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
) -> None:
    """Load model weights and strip the '_orig_mod.' prefix from torch.compile."""
    state_dict = torch.load(path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

    if missing:
        print(f"Missing keys: {len(missing)}")
        print("   first 10:", missing[:10])
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        print("   first 10:", unexpected[:10])
    if missing or unexpected:
        print(
            "State-dict mismatch detected. This usually means that the config "
            "does not match the saved checkpoint, such as hidden_dim, dropout, "
            "or Phase 2 module initialization."
        )


def infer_aligner_path(collusion_pkl: str) -> str:
    """Infer aligner path from a Phase 2 collusion-graph pickle path."""
    base = os.path.basename(collusion_pkl)
    dirname = os.path.dirname(collusion_pkl)
    if base.startswith('phase2_collusion'):
        return os.path.join(dirname, base.replace('phase2_collusion', 'aligner', 1))
    return os.path.join(dirname, 'aligner.pkl')


def load_threshold(checkpoint: str, manual_threshold: Optional[float]) -> float:
    """Load the validation-selected threshold or use a manual fallback."""
    if manual_threshold is not None:
        print(f"Using manual threshold: {manual_threshold:.6f}")
        return float(manual_threshold)

    threshold_json = checkpoint + '.threshold.json'
    if os.path.exists(threshold_json):
        with open(threshold_json, 'r', encoding='utf-8') as f:
            info = json.load(f)
        threshold = float(info.get('threshold', 0.5))
        print(f"Loaded threshold={threshold:.6f} from {threshold_json}")
        print(f"   threshold info: {info}")
        return threshold

    print(
        "No threshold JSON file found. Falling back to threshold=0.5. "
        "Use --threshold if the training run selected a different value."
    )
    return 0.5


def move_collusion_to_device(
    collusion_data: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Move tensor values in collusion_data to the target device."""
    if collusion_data is None:
        return None
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in collusion_data.items()
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print retest metrics in a compact, reproducible format."""
    print("\nRetest results:")
    print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  Macro-F1:          {metrics.get('macro_f1', 0):.4f}")
    print(f"  Precision:         {metrics.get('precision', 0):.4f}")
    print(f"  Recall:            {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:          {metrics.get('f1', 0):.4f}")
    print(f"  AUC:               {metrics.get('auc', 0):.4f}")
    print(f"  Threshold:         {metrics.get('threshold', 0.5):.4f}")
    print(
        "  Prob range:        "
        f"[{metrics.get('prob_min', 0):.4f}, {metrics.get('prob_max', 0):.4f}]"
    )
    print(f"  Prob mean:         {metrics.get('prob_mean', 0):.4f}")
    print(f"  Pred fake rate:    {metrics.get('pred_pos_rate', 0):.4f}")
    print(f"  Samples Real/Fake: {metrics.get('n_real', 0)} / {metrics.get('n_fake', 0)}")
    print("  Confusion Matrix [label: 0=Real, 1=Fake]:")
    print(f"    TN(real->real): {metrics.get('tn', 0)}")
    print(f"    FP(real->fake): {metrics.get('fp', 0)}")
    print(f"    FN(fake->real): {metrics.get('fn', 0)}")
    print(f"    TP(fake->fake): {metrics.get('tp', 0)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Retest a saved MIRAGE checkpoint on the test split.'
    )
    parser.add_argument('--dataset', required=True, choices=['gossipcop', 'politifact'])
    parser.add_argument(
        '--config',
        default='ultra',
        choices=['ultra', 'balanced', 'speed', 'debug', 'pro6000'],
    )
    parser.add_argument('--data_root', default='/path/to/data')
    parser.add_argument(
        '--checkpoint',
        required=True,
        help='Path to the saved Phase 3 checkpoint, e.g. checkpoints/phase3_best_pid_xxx.pth.',
    )
    parser.add_argument(
        '--collusion_pkl',
        required=True,
        help='Path to the saved Phase 2 graph, e.g. checkpoints/phase2_collusion_pid_xxx.pkl.',
    )
    parser.add_argument(
        '--aligner_pkl',
        default=None,
        help='Path to the saved aligner. If omitted, it is inferred from --collusion_pkl.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Manual classification threshold. By default, checkpoint.threshold.json is used.',
    )
    parser.add_argument('--seed', type=int, default=42)

    # These values must match the training run when they affect data or shapes.
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--lambda2', type=float, default=None, help='Override Config.LAMBDA_2.')
    parser.add_argument(
        '--dead_hours',
        type=float,
        default=None,
        help='Override the temporal observation window in hours.',
    )

    parser.add_argument(
        '--main_module',
        default='main',
        help='Module that provides load_data() and build_model().',
    )
    parser.add_argument('--save_json', default=None, help='Optional path to save retest metrics.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")
    if not os.path.exists(args.collusion_pkl):
        raise FileNotFoundError(f"collusion_pkl does not exist: {args.collusion_pkl}")

    aligner_pkl = args.aligner_pkl or infer_aligner_path(args.collusion_pkl)
    if not os.path.exists(aligner_pkl):
        raise FileNotFoundError(
            f"aligner_pkl does not exist: {aligner_pkl}\n"
            "For reproducibility, pass the aligner saved by the same training run."
        )

    set_reproducibility(args.seed)

    main_mod = importlib.import_module(args.main_module)
    from config import get_config
    from training.trainer import MRDETGMTrainer

    Config = get_config(args.config)
    Config.DATASET = args.dataset
    Config.DATA_ROOT = args.data_root
    Config.SEED = args.seed

    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.dropout is not None:
        Config.DROPOUT = args.dropout
    if args.hidden_dim is not None:
        Config.HIDDEN_DIM = args.hidden_dim
    if args.lambda2 is not None:
        Config.LAMBDA_2 = args.lambda2
    if args.dead_hours is not None:
        Config.DEAD_CRITERION_HOURS = args.dead_hours

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("Retest saved checkpoint")
    print("=" * 70)
    print(f"dataset       : {args.dataset}")
    print(f"checkpoint    : {args.checkpoint}")
    print(f"collusion_pkl : {args.collusion_pkl}")
    print(f"aligner_pkl   : {aligner_pkl}")
    print(f"device        : {device}")
    print(f"dead_hours    : {getattr(Config, 'DEAD_CRITERION_HOURS', None)}")
    print(f"hidden_dim    : {getattr(Config, 'HIDDEN_DIM', None)}")
    print(f"dropout       : {getattr(Config, 'DROPOUT', None)}")
    print(f"lambda2       : {getattr(Config, 'LAMBDA_2', None)}")
    print("=" * 70)

    fake_args = argparse.Namespace(
        dataset=args.dataset,
        data_root=args.data_root,
        hparam_json=None,
    )
    _, _, test_loader, _ = main_mod.load_data(fake_args, Config)

    with open(aligner_pkl, 'rb') as f:
        aligner = pickle.load(f)
    twitter_id_map = aligner.user2idx if aligner is not None else {}
    print(f"Loaded aligner: {aligner_pkl} | users={len(twitter_id_map):,}")

    model = main_mod.build_model(Config, device)
    model.initialize_phase2_modules(
        Config,
        device,
        twitter_id_to_global_idx=twitter_id_map,
    )
    load_clean_state_dict(model, args.checkpoint, device)
    model.eval()
    print(f"Loaded Phase 3 checkpoint: {args.checkpoint}")

    with open(args.collusion_pkl, 'rb') as f:
        collusion_data = pickle.load(f)
    collusion_data = move_collusion_to_device(collusion_data, device)
    print(f"Loaded Phase 2 graph: {args.collusion_pkl}")
    if isinstance(collusion_data, dict):
        print("   collusion_data keys:", sorted(collusion_data.keys()))

    threshold = load_threshold(args.checkpoint, args.threshold)
    trainer = MRDETGMTrainer(model, Config, device=str(device))
    trainer.best_threshold = threshold
    metrics = trainer.evaluate(
        test_loader,
        phase=3,
        collusion_data=collusion_data,
        threshold=threshold,
        optimize_threshold=False,
    )
    print_metrics(metrics)

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or '.', exist_ok=True)
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved retest metrics to: {args.save_json}")


if __name__ == '__main__':
    main()
