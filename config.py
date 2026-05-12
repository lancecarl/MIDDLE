"""
MR-DE-TGM configuration.

Defines base hyperparameters and hardware-specific profiles.
"""

import os
import torch


# Base config

class BaseConfig:
    """Base hyperparameters shared by all configs."""

    # ── Data ────────────────────────────────────────────────────────────────
    DATA_ROOT    = '/path/to/data'
    FEATURE_TYPE = 'bert'       # 'bert' | 'spacy' | 'content'
    DATASET      = 'gossipcop'  # 'gossipcop' | 'politifact'

    # ── Training ────────────────────────────────────────────────────────────
    SEED         = 42
    BATCH_SIZE   = 32
    NUM_EPOCHS   = 100          # Legacy compatibility; use PHASE1_EPOCHS / PHASE3_EPOCHS
    LEARNING_RATE= 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 15

    # ── Model structure ─────────────────────────────────────────────────────
    HIDDEN_DIM        = 128
    PROP_DIM          = 128
    USER_DIM          = 64
    NUM_HEADS         = 4
    NUM_LAYERS        = 2
    DROPOUT           = 0.3
    FUSION_TYPE       = 'concat'   # 'concat' | 'gate'
    # ── Device / performance ──────────────────────────────────────────────
    DEVICE                      = 'cuda'
    NUM_WORKERS                 = 4
    PIN_MEMORY                  = True
    PREFETCH_FACTOR             = 2
    PERSISTENT_WORKERS          = True
    USE_AMP                     = True
    GRADIENT_ACCUMULATION_STEPS = 2
    MAX_GRAD_NORM               = 1.0
    EMPTY_CACHE_FREQ            = 50
    LABEL_SMOOTHING             = 0.0
    USE_TORCH_COMPILE           = False
    COMPILE_MODE                = 'default'
    COMPILE_FULLGRAPH           = False

    @classmethod
    def apply_optimizations(cls):
        """Apply environment-level optimizations."""
        torch.backends.cudnn.benchmark    = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
        os.environ.setdefault('OMP_NUM_THREADS', '8')
        os.environ.setdefault('MKL_NUM_THREADS', '8')


# MR-DE-TGM config

class MRDETGMConfig(BaseConfig):
    """Full hyperparameter set for MR-DE-TGM."""

    # ── Temporal context ────────────────────────────────────────────────────
    TEMPORAL_CONTEXT_AWARE = True

    # dead_criterion: observation window per article (hours).
    DEAD_CRITERION_HOURS = 12.0

    # Dataset split ratios (fallback when no custom split exists).
    TRAIN_RATIO = 0.7
    VAL_RATIO   = 0.1

    # ── Loss weights ───────────────────────────────────────────────────────
    LAMBDA_0 = 0.5    # L_init
    LAMBDA_2 = 0.3    # L_MIL
    LAMBDA_3 = 0.05   # L_cons
    LAMBDA_4 = 0      # L_sparse
    LAMBDA_5 = 0.1    # L_balance

    # ── Collusion graph ────────────────────────────────────────────────────
    COLLUSION_EPS             = 3600.0   # Time sync window (seconds)
    COLLUSION_MIN_COOCCURRENCE= 2        # Minimum co-participation count
    COLLUSION_K               = 10       # Soft clusters K
    COLLUSION_HIDDEN          = 128      # Collusion GNN hidden size
    COLLUSION_THRESHOLD       = 0.05     # Edge weight floor

    # ── Stage-aware ─────────────────────────────────────────────────────────
    STAGE_ATTENTION  = True    # Enable stage-aware Transformer attention
    STRUCT_FEAT_DIM  = 7       # Structural feature dim (Delta n/d/w + burst)

    # ── Three-phase training ───────────────────────────────────────────────
    PHASE1_EPOCHS    = 50      # Phase 1 max epochs (pretraining)
    PHASE3_EPOCHS    = 50      # Phase 3 max epochs (finetuning)
    PHASE2_THRESHOLD = 0.3     # High-risk threshold for p_hat weighting

    # ── UserRiskMILEncoder ────────────────────────────────────────────────
    USER_RISK_HIDDEN  = 64     # User risk GNN hidden size
    USER_RISK_LAYERS  = 2      # User risk GNN layers
    MIL_AGGREGATION   = 'noisy-or'  # MIL aggregation: 'noisy-or' | 'mean'

    # ── Performance overrides ─────────────────────────────────────────────
    PATIENCE              = 15
    GRADIENT_ACCUMULATION_STEPS = 2
    USE_AMP               = True


# RTX 4090 optimized profile

class RTX4090UltraConfig(MRDETGMConfig):
    """RTX 4090 / 24GB tuned profile."""

    # ── Model dimensions ───────────────────────────────────────────────────
    HIDDEN_DIM          = 512
    PROP_DIM            = 512
    PROP_HIDDEN         = 512
    TRUST_DIM           = 256
    TRUST_HIDDEN        = 256
    FUSION_HIDDEN       = 512
    COLLUSION_HIDDEN    = 256
    COLLUSION_K         = 16
    GROUP_DIM           = 128

    # ── Training ───────────────────────────────────────────────────────────
    BATCH_SIZE                  = 128
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE               = 5e-4   # Lower LR for larger model
    MAX_GRAD_NORM               = 1.0

    # ── DataLoader ─────────────────────────────────────────────────────────
    NUM_WORKERS                 = 8
    PREFETCH_FACTOR             = 4
    PIN_MEMORY                  = True
    PERSISTENT_WORKERS          = True

    # ── Mixed precision ────────────────────────────────────────────────────
    USE_AMP                     = True
    EMPTY_CACHE_FREQ            = 200

    # ── Compile (disabled) ────────────────────────────────────────────────
    USE_TORCH_COMPILE           = False

    # ── Epochs ─────────────────────────────────────────────────────────────
    PHASE1_EPOCHS               = 150
    PHASE3_EPOCHS               = 150
    PATIENCE                    = 20

    @classmethod
    def apply_optimizations(cls):
        super().apply_optimizations()
        # Allow larger allocation chunks for 24GB VRAM.
        os.environ.setdefault(
            'PYTORCH_CUDA_ALLOC_CONF',
            'max_split_size_mb:512'
        )


class ProW6000BlackwellConfig(MRDETGMConfig):
    """RTX Pro 6000 Blackwell profile."""

    BATCH_SIZE                  = 256
    GRADIENT_ACCUMULATION_STEPS = 1

    NUM_WORKERS                 = 16
    PREFETCH_FACTOR             = 8
    PIN_MEMORY                  = True
    PERSISTENT_WORKERS          = True

    USE_AMP                     = True

    USE_TORCH_COMPILE           = False
    COMPILE_MODE                = 'max-autotune'
    COMPILE_FULLGRAPH           = False

    EMPTY_CACHE_FREQ            = 500

    PHASE1_EPOCHS               = 100
    PHASE3_EPOCHS               = 100
    PATIENCE                    = 15

    MAX_GRAD_NORM               = 2.0
    LEARNING_RATE               = 8e-3

    @classmethod
    def apply_optimizations(cls):
        """Extend parent optimizations with Blackwell-specific settings."""
        super().apply_optimizations()

        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        os.environ['OMP_NUM_THREADS']  = '16'
        os.environ['MKL_NUM_THREADS']  = '16'

        os.environ.setdefault(
            'PYTORCH_CUDA_ALLOC_CONF',
            'max_split_size_mb:512'
        )


# Debug/CPU profile

class DebugConfig(MRDETGMConfig):
    """Quick debug config (CPU friendly, small batch, few epochs)."""

    DEVICE                      = 'cpu'
    BATCH_SIZE                  = 4
    NUM_WORKERS                 = 0
    PIN_MEMORY                  = False
    PREFETCH_FACTOR             = None
    PERSISTENT_WORKERS          = False
    USE_AMP                     = False
    GRADIENT_ACCUMULATION_STEPS = 1
    USE_TORCH_COMPILE           = False

    PHASE1_EPOCHS = 3
    PHASE3_EPOCHS = 5
    PATIENCE      = 3

    COLLUSION_K               = 3
    COLLUSION_MIN_COOCCURRENCE= 1

    # Disable temporal filter for faster debugging if needed.
    TEMPORAL_CONTEXT_AWARE    = False


# Config registry

CONFIG_REGISTRY = {
    'ultra':    RTX4090UltraConfig,
    'balanced': MRDETGMConfig,
    'speed':    RTX4090UltraConfig,
    'debug':    DebugConfig,
    'pro6000':  ProW6000BlackwellConfig,
}


def get_config(name: str = 'balanced'):
    """Fetch a config class by name."""
    name = name.lower()
    if name not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown config '{name}'. Options: {list(CONFIG_REGISTRY.keys())}"
        )
    return CONFIG_REGISTRY[name]
