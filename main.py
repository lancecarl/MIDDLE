"""
MR-DE-TGM entrypoint.

Runs phase1/phase2/phase3 training or GPU monitoring.
"""

from __future__ import annotations

import argparse
import os
import warnings
import pickle
from typing import Optional, Dict, Any

import torch

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────────────
from config import get_config, RTX4090UltraConfig


# Helper: load weights safely (strip torch.compile prefix)
def load_clean_state_dict(model: torch.nn.Module, path: str, device: torch.device):
    """Load weights and strip the '_orig_mod.' prefix from torch.compile."""
    state_dict = torch.load(path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)

def get_unique_id(args) -> str:
    """Generate a unique suffix to avoid multi-process file collisions."""
    if getattr(args, 'hparam_json', None):
        # Extract JSON file stem, e.g., 'trial_12'
        return os.path.splitext(os.path.basename(args.hparam_json))[0]
    return f"pid_{os.getpid()}"
# Environment setup

def setup_environment(Config) -> torch.device:
    """Apply optimizations, set seeds, and create output directories."""
    print("\n" + "=" * 70)
    print("Initializing environment")
    print("=" * 70)

    Config.apply_optimizations()

    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)

    for d in ('checkpoints', 'results', 'visualizations', 'logs'):
        os.makedirs(d, exist_ok=True)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f"CUDA:   {torch.version.cuda}")
        print(f"PyTorch: {torch.__version__}")
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")

    print("=" * 70 + "\n")
    return device


# Shared data loader

def load_data(args, Config):
    """Load data and return (train_loader, val_loader, test_loader, loader_obj)."""
    from utils.enhanced_data_loader import TemporalUPFDDataLoaderOptimized

    # Load split and temporal settings from Config with fallback defaults.
    train_ratio            = getattr(Config, 'TRAIN_RATIO',            0.7)
    val_ratio              = getattr(Config, 'VAL_RATIO',              0.1)
    temporal_context_aware = getattr(Config, 'TEMPORAL_CONTEXT_AWARE', True)
    dead_criterion_hours   = getattr(Config, 'DEAD_CRITERION_HOURS',   12.0)

    print("Loading dataset...")
    print(f"  Split: train={train_ratio:.0%} / val={val_ratio:.0%} / "
          f"test={1-train_ratio-val_ratio:.0%}")
    print("  Temporal filter: "
          f"{'enabled' if temporal_context_aware else 'disabled'}"
          + (f" | dead_criterion={dead_criterion_hours:.0f}h" if temporal_context_aware else ""))

    loader_obj = TemporalUPFDDataLoaderOptimized(
        root=Config.DATA_ROOT,
        name=args.dataset,
        feature=Config.FEATURE_TYPE,
        use_custom_split=True,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        temporal_context_aware=temporal_context_aware,
        dead_criterion_hours=dead_criterion_hours,
    )

    train_loader, val_loader, test_loader = loader_obj.get_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR
            if Config.NUM_WORKERS > 0 else None,
        persistent_workers=Config.PERSISTENT_WORKERS
            if Config.NUM_WORKERS > 0 else False,
    )

    print(f"  train: {len(train_loader.dataset)} samples")
    print(f"  val:   {len(val_loader.dataset)} samples")
    print(f"  test:  {len(test_loader.dataset)} samples")
    return train_loader, val_loader, test_loader, loader_obj


def build_model(Config, device):
    """Build MRDETGMModel and print parameter count."""
    from models.integrated_model import MRDETGMModel

    print("\nBuilding model...")
    model = MRDETGMModel(Config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f} M")
    return model.to(device)


def load_twitter_aligner(loader_obj, Config):
    """Build TwitterIDAligner with train-only user indexing."""
    try:
        from utils.user_alignment import build_aligner_from_loader

        # Train-only indexing prevents leakage from non-train users.
        aligner = build_aligner_from_loader(loader_obj, train_only=True)

        aligner.build_participation_matrix()
        aligner.compute_idf()
        print("TwitterIDAligner initialized (train_only=True)")
        return aligner
    except Exception as exc:
        print(f"TwitterIDAligner init failed: {exc}. Skip collusion graph build.")
        return None


def print_test_results(metrics: Dict[str, float]):
    """Print test metrics."""
    print("\nTest results:")
    print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  Macro-F1:          {metrics.get('macro_f1', 0):.4f}")
    print(f"  Precision:         {metrics.get('precision', 0):.4f}")
    print(f"  Recall:            {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:          {metrics.get('f1', 0):.4f}")
    print(f"  AUC:               {metrics.get('auc', 0):.4f}")
    print(f"  Threshold:         {metrics.get('threshold', 0.5):.4f}")
    print(f"  Prob range:        [{metrics.get('prob_min', 0):.4f}, {metrics.get('prob_max', 0):.4f}]")
    print(f"  Prob mean:         {metrics.get('prob_mean', 0):.4f}")
    print(f"  Pred fake rate:    {metrics.get('pred_pos_rate', 0):.4f}")
    print(f"  Samples Real/Fake: {metrics.get('n_real', 0)} / {metrics.get('n_fake', 0)}")
    print("  Confusion Matrix [label: 0=Real, 1=Fake]:")
    print(f"    TN(real->real): {metrics.get('tn', 0)}")
    print(f"    FP(real->fake): {metrics.get('fp', 0)}")
    print(f"    FN(fake->real): {metrics.get('fn', 0)}")
    print(f"    TP(fake->fake): {metrics.get('tp', 0)}")


# Phase 1: event detection pretraining

def train_phase1_only(args, Config=None):
    """Run Phase 1 pretraining and save best weights and p_hat_0."""
    if Config is None:
        Config = get_config(args.config)

    device = setup_environment(Config)

    train_loader, val_loader, test_loader, loader_obj = load_data(args, Config)
    model = build_model(Config, device)

    from training.trainer import MRDETGMTrainer
    trainer = MRDETGMTrainer(model, Config, device=str(device))

    print("\nPhase 1 training started...")
    p1_metrics = trainer.train_phase1(
        train_loader, val_loader,
        num_epochs=Config.PHASE1_EPOCHS,
        save_path='checkpoints/phase1_best.pth',
    )

    load_clean_state_dict(model, 'checkpoints/phase1_best.pth', device)

    test_metrics = trainer.evaluate(test_loader, phase=1)
    print("\nPhase 1 test evaluation:")
    print_test_results(test_metrics)

    _save_training_curves(trainer, 'visualizations/phase1_curves.png')

    return trainer, model, loader_obj


# Phase 2: offline graph construction

def build_graphs(args, Config=None,
                 trainer=None, model=None, loader_obj=None):
    """Load Phase 1 predictions and build the Phase 2 collusion graph."""
    if Config is None:
        Config = get_config(args.config)

    if model is None:
        device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader, _, loader_obj = load_data(args, Config)
        model = build_model(Config, device)

        load_clean_state_dict(model, 'checkpoints/phase1_best.pth', device)

        from training.trainer import MRDETGMTrainer
        trainer = MRDETGMTrainer(model, Config, device=str(device))
    else:
        device = next(model.parameters()).device
        train_loader, val_loader, _, _ = load_data(args, Config)

    twitter_aligner = load_twitter_aligner(loader_obj, Config)

    print("\nPhase 2: building graph structures...")
    collusion_data = trainer.build_phase2_graphs(
        full_dataset=None,
        twitter_aligner=twitter_aligner,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    save_dict = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in collusion_data.items()
        if k != 'global_z_u' or v is not None
    }
    with open('checkpoints/phase2_collusion.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
    print("Phase 2 graph data saved to checkpoints/phase2_collusion.pkl")

    return collusion_data


# Phase 3: joint finetuning

def train_phase3_finetune(args, Config=None,
                          collusion_data=None,
                          trainer=None, model=None,
                          loader_obj=None):
    """Load Phase 2 data and run Phase 3 joint finetuning."""
    if Config is None:
        Config = get_config(args.config)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

    if model is None:
        train_loader, val_loader, test_loader, loader_obj = load_data(args, Config)
        model = build_model(Config, device)
        p1_ckpt = 'checkpoints/phase1_best.pth'
        if os.path.exists(p1_ckpt):
            load_clean_state_dict(model, p1_ckpt, device)
            print(f"Loaded Phase 1 weights: {p1_ckpt}")

        from training.trainer import MRDETGMTrainer
        trainer = MRDETGMTrainer(model, Config, device=str(device))
    else:
        train_loader, val_loader, test_loader, _ = load_data(args, Config)

    if collusion_data is None:
        pkl_path = 'checkpoints/phase2_collusion.pkl'
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                collusion_data = pickle.load(f)
            print(f"Loaded Phase 2 graph data: {pkl_path}")
        else:
            print("Phase 2 data not found; Phase 3 will fall back to Phase 1 mode")
            collusion_data = None

    print("\nPhase 3 training started...")
    p3_metrics = trainer.train_phase3(
        train_loader, val_loader,
        num_epochs=Config.PHASE3_EPOCHS,
        collusion_data=collusion_data,
        save_path='checkpoints/phase3_best.pth',
    )

    load_clean_state_dict(model, 'checkpoints/phase3_best.pth', device)

    test_metrics = trainer.evaluate(
        test_loader, phase=3, collusion_data=collusion_data
    )
    print("\nPhase 3 test evaluation:")
    print_test_results(test_metrics)

    _save_training_curves(trainer, 'visualizations/phase3_curves.png')

    return p3_metrics, test_metrics


# Full pipeline
def train_all(args, Config=None):
    """Run Phase 1 → Phase 2 → Phase 3 sequentially."""
    if Config is None:
        Config = get_config(args.config)

    device = setup_environment(Config)

    # Use a per-trial suffix to avoid collisions.
    uid = get_unique_id(args)
    p1_ckpt = f'checkpoints/phase1_best_{uid}.pth'
    p2_data = f'checkpoints/phase2_collusion_{uid}.pkl'
    p3_ckpt = f'checkpoints/phase3_best_{uid}.pth'

    train_loader, val_loader, test_loader, loader_obj = load_data(args, Config)
    model = build_model(Config, device)

    from training.trainer import MRDETGMTrainer
    trainer = MRDETGMTrainer(model, Config, device=str(device))
    twitter_aligner = load_twitter_aligner(loader_obj, Config)

    print("\n" + "=" * 70)
    print("MR-DE-TGM full training (Phase 1 -> 2 -> 3)")
    print("=" * 70)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    p1_metrics = trainer.train_phase1(
        train_loader, val_loader,
        num_epochs=Config.PHASE1_EPOCHS,
        save_path=p1_ckpt,
    )

    load_clean_state_dict(model, p1_ckpt, device)

    print(f"\nPhase 1 done: best AUC={p1_metrics.get('auc', 0):.4f}  "
          f"F1={p1_metrics.get('f1', 0):.4f}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    collusion_data = trainer.build_phase2_graphs(
        full_dataset=None,
        twitter_aligner=twitter_aligner,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    save_dict = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in collusion_data.items()
        if k != 'global_z_u' or v is not None
    }
    with open(p2_data, 'wb') as f:
            pickle.dump(save_dict, f)

    # Save aligner for consistent user ordering if needed by analysis.
    if twitter_aligner is not None:
        aligner_path = p2_data.replace('phase2_collusion', 'aligner')
        with open(aligner_path, 'wb') as f:
            pickle.dump(twitter_aligner, f)
        print(f"TwitterIDAligner saved to {aligner_path}")

    # Inject Phase 2 modules before Phase 3.
    twitter_id_map = twitter_aligner.user2idx if twitter_aligner is not None else {}
    model.initialize_phase2_modules(Config, device, twitter_id_to_global_idx=twitter_id_map)

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    p3_metrics = trainer.train_phase3(
        train_loader, val_loader,
        num_epochs=Config.PHASE3_EPOCHS,
        collusion_data=collusion_data,
        save_path=p3_ckpt,
    )

    load_clean_state_dict(model, p3_ckpt, device)

    test_metrics = trainer.evaluate(
        test_loader, phase=3, collusion_data=collusion_data
    )

    print("\n" + "=" * 70)
    print("Final test results (Phase 3):")
    print_test_results(test_metrics)
    print("=" * 70)

    _save_training_curves(trainer, 'visualizations/full_training_curves.png')

    return {
        'phase1': p1_metrics,
        'phase3': p3_metrics,
        'test':   test_metrics,
    }


# GPU monitor

def monitor_gpu():
    """Monitor GPU usage (Ctrl+C to stop)."""
    import subprocess
    import time

    print("\n" + "=" * 70)
    print("GPU monitor (Ctrl+C to stop)")
    print("=" * 70)

    try:
        while True:
            result = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,utilization.memory,'
                 'memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = [s.strip() for s in result.stdout.strip().split(',')]
                gpu_util, mem_util, mem_used, mem_total, temp = parts[:5]
                print(
                    f"\rGPU: {gpu_util}%  "
                    f"VRAM: {mem_util}% ({mem_used}/{mem_total} MB)  "
                    f"Temp: {temp}C",
                    end='', flush=True
                )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped")


# Helpers

def _save_training_curves(trainer, save_path: str):
    """Save training curves (optional, requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        if trainer.history['train_loss']:
            axes[0].plot(trainer.history['train_loss'], label='train')
        if trainer.history['val_loss']:
            axes[0].plot(trainer.history['val_loss'], label='val')
        axes[0].set_title('Loss')
        axes[0].legend()

        if trainer.history['val_acc']:
            axes[1].plot(trainer.history['val_acc'])
        axes[1].set_title('Val Accuracy')

        if trainer.history['val_auc']:
            axes[2].plot(trainer.history['val_auc'])
        axes[2].set_title('Val AUC')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nTraining curves saved: {save_path}")
    except Exception as exc:
        print(f"\nVisualization failed: {exc}. Skipping.")


# CLI

def main():
    parser = argparse.ArgumentParser(
        description='MR-DE-TGM fake news detection (three-stage training)'
    )

    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'monitor'],
        help='Run mode: train | monitor',
    )
    parser.add_argument(
        '--phase', type=str, default='all',
        choices=['phase1', 'phase2', 'phase3', 'all'],
        help=(
            'phase1: run Phase 1 pretraining\n'
            'phase2: run Phase 2 graph build (requires phase1)\n'
            'phase3: run Phase 3 finetuning (requires phase1+2)\n'
            'all:    run all phases (default)'
        ),
    )
    parser.add_argument(
        '--dataset', type=str, default='gossipcop',
        choices=['politifact', 'gossipcop'],
        help='Dataset name',
    )
    parser.add_argument(
        '--config', type=str, default='balanced',
        choices=['ultra', 'balanced', 'speed', 'debug', 'pro6000'],
        help=(
            'ultra:    RTX 4090 optimized\n'
            'balanced: standard config (default)\n'
            'speed:    same as ultra\n'
            'debug:    quick debug (CPU ok)\n'
            'pro6000:  RTX Pro 6000 Blackwell'
        ),
    )

    # Optional overrides
    parser.add_argument('--lr',         type=float, default=None, help='Override learning rate')
    parser.add_argument('--batch_size', type=int,   default=None, help='Override batch size')
    parser.add_argument('--p1_epochs',  type=int,   default=None, help='Override Phase 1 epochs')
    parser.add_argument('--p3_epochs',  type=int,   default=None, help='Override Phase 3 epochs')
    parser.add_argument('--seed',       type=int,   default=None, help='Override random seed')
    parser.add_argument('--hparam_json', type=str,  default=None,
                        help='Hyperparameter override JSON (from grid_search.py)')
    parser.add_argument('--dropout',     type=float, default=None, help='Override dropout')
    parser.add_argument('--hidden_dim',  type=int,   default=None, help='Override hidden dim')
    parser.add_argument('--lambda2',     type=float, default=None, help='Override MIL weight')
    parser.add_argument('--dead_hours',  type=int,   default=None, help='Override dead_criterion_hours')

    args = parser.parse_args()

    # ── Load config and apply overrides ─────────────────────────────────────
    Config = get_config(args.config)

    # ── hparam_json overrides (from grid_search.py) ─────────────────────────
    if args.hparam_json:
        import json, pathlib
        _overrides = json.loads(pathlib.Path(args.hparam_json).read_text())
        for _k, _v in _overrides.items():
            if hasattr(Config, _k):
                setattr(Config, _k, _v)
        if 'LEARNING_RATE' in _overrides: Config.LEARNING_RATE = _overrides['LEARNING_RATE']
        if 'PHASE1_EPOCHS' in _overrides: Config.PHASE1_EPOCHS = _overrides['PHASE1_EPOCHS']
        if 'PHASE3_EPOCHS' in _overrides: Config.PHASE3_EPOCHS = _overrides['PHASE3_EPOCHS']
        if 'DROPOUT'       in _overrides: Config.DROPOUT       = _overrides['DROPOUT']

    # CLI overrides (lower priority than hparam_json).
    if args.lr         is not None: Config.LEARNING_RATE = args.lr
    if args.batch_size is not None: Config.BATCH_SIZE    = args.batch_size
    if args.p1_epochs  is not None: Config.PHASE1_EPOCHS = args.p1_epochs
    if args.p3_epochs  is not None: Config.PHASE3_EPOCHS = args.p3_epochs
    if args.seed       is not None: Config.SEED          = args.seed
    if args.dropout    is not None: Config.DROPOUT                = args.dropout
    if args.hidden_dim is not None: Config.HIDDEN_DIM             = args.hidden_dim
    if args.lambda2    is not None: Config.LAMBDA_2               = args.lambda2
    if args.dead_hours is not None: Config.DEAD_CRITERION_HOURS   = args.dead_hours

    # ── Startup info ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MR-DE-TGM fake news detection")
    print(f"  mode: {args.mode}  phase: {args.phase}  "
          f"dataset: {args.dataset}  config: {args.config}")
    print("  temporal context: "
          f"{'enabled' if getattr(Config, 'TEMPORAL_CONTEXT_AWARE', True) else 'disabled'}")
    print("=" * 70)

    # ── Dispatch ───────────────────────────────────────────────────────────
    if args.mode == 'monitor':
        monitor_gpu()
        return

    if args.phase == 'all':
        train_all(args, Config)

    elif args.phase == 'phase1':
        train_phase1_only(args, Config)

    elif args.phase == 'phase2':
        build_graphs(args, Config)

    elif args.phase == 'phase3':
        train_phase3_finetune(args, Config)
    uid = get_unique_id(args)
    if args.phase == 'all':
        print(f"  Phase 1 best:   checkpoints/phase1_best_{uid}.pth")
        print(f"  Phase 2 data:   checkpoints/phase2_collusion_{uid}.pkl")
        print(f"  Phase 2 aligner: checkpoints/aligner_{uid}.pkl")
        print(f"  Phase 3 best:   checkpoints/phase3_best_{uid}.pth")
        print("  Curves:         visualizations/full_training_curves.png")
    elif args.phase == 'phase1':
        print("  Phase 1 best:   checkpoints/phase1_best.pth")
        print("  Phase 1 curves: visualizations/phase1_curves.png")
    elif args.phase == 'phase2':
        print("  Phase 2 data:   checkpoints/phase2_collusion.pkl")
    elif args.phase == 'phase3':
        print("  Phase 3 best:   checkpoints/phase3_best.pth")
        print("  Phase 3 curves: visualizations/phase3_curves.png")

    print("\nAll tasks finished.\n")
