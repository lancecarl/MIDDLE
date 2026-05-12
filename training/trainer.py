"""Three-stage trainer for MR-DE-TGM."""

from __future__ import annotations

import os
import json
import time
import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
)

# ── Project loss utilities ──────────────────────────────────────────────────
try:
    from utils.mil_loss import (
        mil_loss, consistency_loss, sparse_regularization, balance_loss
    )
    _HAS_MIL_LOSS = True
except ImportError:
    _HAS_MIL_LOSS = False


# Helper: default loss weights when missing in config

_LAMBDA_DEFAULTS = dict(
    LAMBDA_0=0.5,   # L_init
    LAMBDA_2=0.3,   # L_MIL
    LAMBDA_3=0.2,   # L_cons
    LAMBDA_4=1e-4,  # L_sparse
    LAMBDA_5=0.1,   # L_balance
)


def _cfg(config, key: str, default=None):
    """Read config attribute/dict key safely with a default fallback."""
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict) and key in config:
        return config[key]
    if default is None and key in _LAMBDA_DEFAULTS:
        return _LAMBDA_DEFAULTS[key]
    return default


# MRDETGMTrainer — three-stage trainer

class MRDETGMTrainer:
    """
    Three-phase trainer for MR-DE-TGM.

    Typical usage:
    - train_phase1()
    - build_phase2_graphs()
    - train_phase3()
    """

    def __init__(self, model: nn.Module, config, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ── Model ───────────────────────────────────────────────────────────
        # Optional PyTorch 2.0 compile
        if _cfg(config, 'USE_TORCH_COMPILE', False):
            try:
                mode = _cfg(config, 'COMPILE_MODE', 'default')
                model = torch.compile(model, mode=mode,
                                      fullgraph=_cfg(config, 'COMPILE_FULLGRAPH', False))
                print(f"torch.compile ready (mode={mode})")
            except Exception as exc:
                print(f"torch.compile failed: {exc}. Using eager model.")

        self.model = model.to(self.device)

        # ── Loss weights ────────────────────────────────────────────────────
        self.lambda_0 = _cfg(config, 'LAMBDA_0')  # L_init
        self.lambda_2 = _cfg(config, 'LAMBDA_2')  # L_MIL
        self.lambda_3 = _cfg(config, 'LAMBDA_3')  # L_cons
        self.lambda_4 = _cfg(config, 'LAMBDA_4')  # L_sparse
        self.lambda_5 = _cfg(config, 'LAMBDA_5')  # L_balance

        # ── Optimizer (lazy init via _reset_optimizer) ─────────────────────
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

        # ── AMP ────────────────────────────────────────────────────────────
        self.use_amp: bool = _cfg(config, 'USE_AMP', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # ── Gradient handling ──────────────────────────────────────────────
        self.grad_accum_steps: int = _cfg(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.max_grad_norm: float  = _cfg(config, 'MAX_GRAD_NORM', 1.0)
        self.empty_cache_freq: int = _cfg(config, 'EMPTY_CACHE_FREQ', 100)

        # ── Phase 2 threshold ──────────────────────────────────────────────
        self.phase2_threshold: float = _cfg(config, 'PHASE2_THRESHOLD', 0.5)

        # ── Training history ───────────────────────────────────────────────
        self.history: Dict[str, List] = {
            'train_loss': [], 'val_loss': [],
            'val_acc': [], 'val_f1': [], 'val_auc': [],
        }
        self.best_val_auc: float = 0.0
        self.best_val_acc: float = 0.0
        self.best_val_f1:  float = 0.0
        self.best_threshold: float = 0.5
        self.best_selection_score: float = 0.0
        self.best_val_bal_acc: float = 0.0
        self.best_val_macro_f1: float = 0.0

        # Backward compatibility
        self.train_losses = self.history['train_loss']
        self.val_losses   = self.history['val_loss']
        self.val_accs     = self.history['val_acc']
        self.val_f1s      = self.history['val_f1']

        self._print_init()

    # Internal helpers

    def _print_init(self):
        print("\nMRDETGMTrainer initialized")
        print(f"  device:    {self.device}")
        print(f"  amp:       {self.use_amp}")
        print(f"  grad accum:{self.grad_accum_steps}")
        print(f"  loss wts:  λ0={self.lambda_0} λ2={self.lambda_2} "
              f"λ3={self.lambda_3} λ4={self.lambda_4} λ5={self.lambda_5}")

    def _reset_optimizer(self, lr: Optional[float] = None):
        """Reset optimizer and scheduler at each phase start."""
        lr = lr or _cfg(self.config, 'LEARNING_RATE', 1e-3)
        wd = _cfg(self.config, 'WEIGHT_DECAY', 5e-4)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=wd, betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )

    def _step_optimizer(self, loss: torch.Tensor, batch_idx: int):
        """Handle grad accumulation, AMP, clipping, and optimizer steps."""
        scaled = loss / self.grad_accum_steps

        if self.use_amp:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

    # Loss computation

    def compute_total_loss(
        self,
        p_hat:   torch.Tensor,   # [B] final prediction
        p_user:  torch.Tensor,   # [B] MIL user prediction
        p_hat_0: torch.Tensor,   # [B] initial prediction
        y:       torch.Tensor,   # [B] labels (int)
        M_U:     Optional[torch.Tensor] = None,  # user graph mask
        S:       Optional[torch.Tensor] = None,  # soft assignments [U, K]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate loss terms and return total and components."""
        # Force float32 to avoid AMP dtype mismatches.
        p_hat   = p_hat.to(torch.float32)
        p_user  = p_user.to(torch.float32)
        p_hat_0 = p_hat_0.to(torch.float32)
        y_f     = y.to(torch.float32)

        # Clamp probabilities for numerical stability.
        eps = 1e-7
        p_hat   = p_hat.clamp(eps, 1 - eps)
        p_user  = p_user.clamp(eps, 1 - eps)
        p_hat_0 = p_hat_0.clamp(eps, 1 - eps)

        # Core losses
        L_event = F.binary_cross_entropy(p_hat,   y_f)
        L_init  = F.binary_cross_entropy(p_hat_0, y_f)
        L_MIL   = F.binary_cross_entropy(p_user,  y_f)

        losses = {
            'event': L_event,
            'init':  self.lambda_0 * L_init,
            'mil':   self.lambda_2 * L_MIL,
        }

        # Sparse regularization
        if M_U is not None and _HAS_MIL_LOSS:
            losses['sparse'] = sparse_regularization(M_U, self.lambda_4)
        elif M_U is not None:
            losses['sparse'] = self.lambda_4 * M_U.abs().mean()

        # Balance regularization
        if S is not None and _HAS_MIL_LOSS:
            losses['balance'] = self.lambda_5 * balance_loss(S)
        elif S is not None:
            # Fallback: variance penalty
            mean_assign = S.mean(dim=0)            # [K]
            target = torch.ones_like(mean_assign) / S.size(1)
            losses['balance'] = self.lambda_5 * ((mean_assign - target) ** 2).sum()

        total = sum(losses.values())
        return total, losses

    # Phase 1: event detection pretraining

    def train_phase1(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        save_path: str = 'checkpoints/phase1_best.pth',
        lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """Phase 1 trains only the event detection path."""
        print("\n" + "=" * 70)
        print("Phase 1 pretraining")
        print("=" * 70)

        self._reset_optimizer(lr)
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        patience     = _cfg(self.config, 'PATIENCE', 10)
        patience_cnt = 0
        best_metrics: Dict[str, float] = {}

        for epoch in range(num_epochs):
            print(f"\n📊 [Phase1] Epoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            train_info = self._train_epoch_phase1(train_loader, epoch)
            val_info   = self.evaluate(val_loader, phase=1)

            self.history['train_loss'].append(train_info['loss'])
            self.history['val_loss'].append(val_info['loss'])
            self.history['val_acc'].append(val_info['accuracy'])
            self.history['val_f1'].append(val_info['f1'])
            self.history['val_auc'].append(val_info['auc'])

            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]['lr']

            print(f"  Train: loss={train_info['loss']:.4f}  "
                  f"throughput={train_info['throughput']:.1f} samp/s")
            print(f"  Val:   Acc={val_info['accuracy']:.4f}  "
                  f"F1={val_info['f1']:.4f}  AUC={val_info['auc']:.4f}  "
                  f"loss={val_info['loss']:.4f}")
            print(f"  LR: {lr_now:.2e}")

            # Save best by AUC.
            if val_info['auc'] > self.best_val_auc:
                self.best_val_auc = val_info['auc']
                self.best_val_acc = val_info['accuracy']
                self.best_val_f1  = val_info['f1']
                best_metrics = {k: v for k, v in val_info.items()}
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model (ACC: {self.best_val_acc:.4f}, AUC: {self.best_val_auc:.4f})")
                patience_cnt = 0
            else:
                patience_cnt += 1


            if patience_cnt >= patience:
                print(f"\nEarly stopping ({patience} epochs without improvement)")
                break


        print("\n" + "=" * 70)
        print(f"Phase 1 done. Best AUC={self.best_val_auc:.4f}  "
              f"Acc={self.best_val_acc:.4f}  F1={self.best_val_f1:.4f}")
        print("=" * 70)

        return best_metrics

    def _train_epoch_phase1(self, loader, epoch: int) -> Dict[str, float]:
        """Single epoch training for Phase 1."""
        self.model.train()
        total_loss   = 0.0
        num_batches  = 0
        start_time   = time.time()

        pbar = tqdm(loader, desc='Phase1-Train', leave=False)

        for batch_idx, batch_item in enumerate(pbar):
            # Support (batch_data, time_mappings) or batch_data only.
            if isinstance(batch_item, (tuple, list)):
                batch_data, time_mappings = batch_item[0], batch_item[1]
            else:
                batch_data, time_mappings = batch_item, None

            batch_data = batch_data.to(self.device, non_blocking=True)
            y = batch_data.y.long()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    _, _, p_hat_0_tmp, p_user, p_hat_0 = \
                        self._forward_phase1(batch_data, time_mappings)

                # Compute loss in FP32 outside autocast.
                loss, loss_detail = self.compute_total_loss(
                    p_hat_0, p_user, p_hat_0, y
                )
            else:
                _, _, _, p_user, p_hat_0 = \
                    self._forward_phase1(batch_data, time_mappings)
                loss, loss_detail = self.compute_total_loss(
                    p_hat_0, p_user, p_hat_0, y
                )

            self._step_optimizer(loss, batch_idx)

            total_loss  += loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'event': f'{loss_detail["event"].item():.4f}',
                'mil':   f'{loss_detail["mil"].item():.4f}',
            })

            if (batch_idx + 1) % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        n_samples = len(loader.dataset) if hasattr(loader, 'dataset') else num_batches

        return {
            'loss':       total_loss / max(num_batches, 1),
            'epoch_time': elapsed,
            'throughput': n_samples / elapsed,
        }

    def _forward_phase1(self, batch_data, time_mappings):
        """Call model.forward_phase1 and handle legacy interfaces."""
        if hasattr(self.model, 'forward_phase1'):
            return self.model.forward_phase1(batch_data, time_mappings)
        else:
            # Legacy fallback: forward returns (logits, _, _)
            out = self.model(batch_data, time_mappings)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                p_hat, p_user, p_hat_0 = out
                # Build a compatible 5-tuple
                return None, None, None, p_user, p_hat_0
            raise RuntimeError(
                "Model lacks forward_phase1 and does not return a 3-tuple."
            )


    @staticmethod
    def _extract_graph_indices_from_batch(batch_data, batch_size: int, fallback_start: int = 0) -> list:
        """Extract true graph indices from a PyG batch."""
        graph_ids = None

        # enhanced_data_loader._temporal_collate_fn saves graph_indices=list[int]
        if hasattr(batch_data, 'graph_indices'):
            graph_ids = getattr(batch_data, 'graph_indices')
        elif hasattr(batch_data, 'graph_idx'):
            graph_ids = getattr(batch_data, 'graph_idx')

        if graph_ids is None:
            return list(range(fallback_start, fallback_start + batch_size))

        if isinstance(graph_ids, torch.Tensor):
            graph_ids = graph_ids.detach().cpu().view(-1).tolist()
        elif isinstance(graph_ids, np.ndarray):
            graph_ids = graph_ids.reshape(-1).tolist()
        elif not isinstance(graph_ids, (list, tuple)):
            graph_ids = [int(graph_ids)]

        graph_ids = [int(x) for x in graph_ids]
        if len(graph_ids) != batch_size:
            print(
                f"  graph_idx count ({len(graph_ids)}) != batch_size ({batch_size}); "
                "falling back to sequential indices."
            )
            return list(range(fallback_start, fallback_start + batch_size))

        return graph_ids

    # Phase 2: offline graph construction (no_grad)

    @torch.no_grad()
    def build_phase2_graphs(
        self,
        full_dataset,               # full dataset (train + val)
        twitter_aligner,            # TwitterIDAligner instance
        train_loader=None,          # optional loaders
        val_loader=None,
    ) -> Dict[str, Any]:
        """Phase 2 offline graph build using Phase 1 predictions only."""
        print("\n" + "=" * 70)
        print("Phase 2 graph build (offline, no_grad)")
        print("=" * 70)

        self.model.eval()

        # Register a hook to capture user_risk_encoder embeddings (e.g., 128-d)
        # so we avoid falling back to raw BERT features.
        self._z_u_hook_buf: Dict[str, Any] = {}

        def _z_u_hook(module, inp, out):
            if isinstance(out, (tuple, list)) and len(out) >= 1:
                self._z_u_hook_buf['z_u'] = out[0].detach().cpu()

        _z_u_handle = None
        if hasattr(self.model, 'user_risk_encoder'):
            _z_u_handle = self.model.user_risk_encoder.register_forward_hook(
                _z_u_hook
            )

        # ── Step 1: collect p_hat_0 ─────────────────────────────────────────
        p_hat_dict:      Dict[int, float]         = {}   # {event_idx: p̂}
        global_z_parts:  Dict[int, torch.Tensor]  = {}   # {twitter_id: z}
        event_cursor = 0

        loaders = []
        if train_loader is not None:
            loaders.append(('train', train_loader))
        if val_loader is not None:
            loaders.append(('val',   val_loader))

        if not loaders:
            # Fallback: iterate over full_dataset
            loaders = [('full', self._make_loader(full_dataset))]

        for split_name, loader in loaders:
            print(f"  Collecting {split_name} p_hat_0...")
            for batch_item in tqdm(loader, desc=f'  p̂ ({split_name})', leave=False):
                if isinstance(batch_item, (tuple, list)):
                    batch_data, time_mappings = batch_item[0], batch_item[1]
                else:
                    batch_data, time_mappings = batch_item, None

                batch_data = batch_data.to(self.device)

                _, _, _, _, p_hat_0 = self._forward_phase1(batch_data, time_mappings)
                # p_hat_0: [B]

                # Use true graph_idx as key.
                p_list = p_hat_0.detach().cpu().tolist()
                graph_ids = self._extract_graph_indices_from_batch(
                    batch_data, batch_size=len(p_list), fallback_start=event_cursor
                )
                for g_idx, p_val in zip(graph_ids, p_list):
                    p_hat_dict[int(g_idx)] = float(p_val)

                # Collect user embeddings (z_u) and aggregate by twitter_ids.
                if hasattr(batch_data, 'twitter_ids') and \
                        twitter_aligner is not None:
                    self._collect_user_embeddings(
                        batch_data, global_z_parts
                    )

                event_cursor += len(p_list)

        print(f"  Collected {event_cursor} samples; "
              f"p_hat_dict covers {len(p_hat_dict)} graph_idx. "
              f"High-risk (p_hat>{self.phase2_threshold})="
              f"{sum(1 for v in p_hat_dict.values() if v > self.phase2_threshold)}")

        # Clear hook and buffer
        if _z_u_handle is not None:
            _z_u_handle.remove()
        self._z_u_hook_buf.clear()

        # ── Step 2: build collusion graph ──────────────────────────────────
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)

        if twitter_aligner is not None:
            try:
                edge_index, edge_weight = \
                    twitter_aligner.build_user_cooccurrence_graph(
                        p_hat_dict,
                        min_cooccurrence=_cfg(
                            self.config, 'COLLUSION_MIN_COOCCURRENCE', 2
                        ),
                    )
                print(f"  Collusion graph built: {edge_index.shape[1]} edges, "
                      f"{edge_weight.shape[0]} weights")
            except Exception as exc:
                print(f"  Collusion graph build failed: {exc}. Using empty graph.")
        else:
            print("  twitter_aligner is None; skip collusion graph")

        # ── Step 3: global user embeddings ─────────────────────────────────
        global_z_u = None
        if global_z_parts:
            # Align to TwitterIDAligner.user_list; fill missing users with zeros.
            if twitter_aligner is not None and hasattr(twitter_aligner, 'user_list'):
                ref_users = list(twitter_aligner.user_list)
            else:
                ref_users = sorted(global_z_parts.keys())

            first_vec = next(iter(global_z_parts.values()))
            zero_vec = torch.zeros_like(first_vec)
            vecs = [global_z_parts.get(uid, zero_vec) for uid in ref_users]
            global_z_u = torch.stack(vecs, dim=0)   # [U, D]
            missing = sum(1 for uid in ref_users if uid not in global_z_parts)
            print(f"  Global user embeddings: {global_z_u.shape} (missing={missing})")

        collusion_data = {
            'edge_index':   edge_index,
            'edge_weight':  edge_weight,
            'global_z_u':   global_z_u,
            'p_hat_dict':   p_hat_dict,
        }

        print("Phase 2 done.\n")
        return collusion_data

    def _collect_user_embeddings(self, batch_data, global_z_parts: dict):
        """Aggregate node embeddings by twitter_id into global_z_parts."""
        if not hasattr(batch_data, 'twitter_ids'):
            return

        t_ids = batch_data.twitter_ids   # [N]
        batch = batch_data.batch         # [N]

        # Root node mask (exclude the news root node per graph).
        root_mask = torch.zeros(batch.size(0), dtype=torch.bool,
                                device=batch.device)
        seen: set = set()
        for idx, g in enumerate(batch.tolist()):
            if g not in seen:
                root_mask[idx] = True
                seen.add(g)

        non_root = ~root_mask

        # Prefer user_risk_encoder output when available; else fallback to batch_data.x.
        feats = getattr(self, '_z_u_hook_buf', {}).get('z_u', None)
        if feats is not None and feats.shape[0] == batch.shape[0]:
            # Hook buffer matches current batch.
            pass
        else:
            # Fallback: raw features; CollusionGraphModule handles projection.
            feats = batch_data.x    # [N, D_raw]

        valid_ids   = t_ids[non_root.to(t_ids.device)].tolist()
        valid_feats = feats[non_root.to(feats.device)]

        for tid, fvec in zip(valid_ids, valid_feats):
            if tid < 0:   # -1 indicates unknown user
                continue
            if tid not in global_z_parts:
                global_z_parts[tid] = fvec.detach().cpu()
            else:
                # Mean aggregation
                global_z_parts[tid] = (
                    global_z_parts[tid] + fvec.detach().cpu()
                ) / 2.0

    @staticmethod
    def _make_loader(dataset, batch_size=32):
        """Temporary DataLoader when no loader is provided."""
        from torch_geometric.loader import DataLoader as PyGLoader
        return PyGLoader(dataset, batch_size=batch_size, shuffle=False)

    # Phase 3: joint finetuning

    def train_phase3(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        collusion_data: Optional[Dict] = None,  # Phase 2 output
        save_path: str = 'checkpoints/phase3_best.pth',
        lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """Phase 3 joint finetuning (full loss set)."""
        print("\n" + "=" * 70)
        print("Phase 3 finetuning")
        print("=" * 70)

        # Prepare Phase 2 data on device.
        if collusion_data is not None:
            collusion_data = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in collusion_data.items()
            }

        # Use a smaller learning rate for Phase 3.
        p3_lr = lr or (_cfg(self.config, 'LEARNING_RATE', 1e-3) * 0.1)
        self._reset_optimizer(p3_lr)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        patience     = _cfg(self.config, 'PATIENCE', 10)
        patience_cnt = 0
        best_metrics: Dict[str, float] = {}

        # Reset best metrics for Phase 3.
        self.best_val_auc = 0.0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_threshold = 0.5
        self.best_selection_score = -1.0
        self.best_val_bal_acc = 0.0
        self.best_val_macro_f1 = 0.0

        for epoch in range(num_epochs):
            print(f"\n📊 [Phase3] Epoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            train_info = self._train_epoch_phase3(
                train_loader, epoch, collusion_data
            )
            val_info = self.evaluate(
                val_loader, phase=3, collusion_data=collusion_data,
                optimize_threshold=True
            )

            self.history['train_loss'].append(train_info['loss'])
            self.history['val_loss'].append(val_info['loss'])
            self.history['val_acc'].append(val_info['accuracy'])
            self.history['val_f1'].append(val_info['f1'])
            self.history['val_auc'].append(val_info['auc'])

            self.scheduler.step()
            lr_now = self.optimizer.param_groups[0]['lr']

            print(f"  Train: loss={train_info['loss']:.4f}  "
                  f"throughput={train_info['throughput']:.1f} samp/s")
            print(f"  Val:   Acc={val_info['accuracy']:.4f}  "
                  f"F1={val_info['f1']:.4f}  BalAcc={val_info.get('balanced_accuracy', 0):.4f}  "
                  f"MacroF1={val_info.get('macro_f1', 0):.4f}  AUC={val_info['auc']:.4f}  "
                  f"Thr={val_info.get('threshold', 0.5):.4f}")
            print(f"  LR: {lr_now:.2e}")

            # Phase 3 model selection uses a composite score.
            bal_acc = float(val_info.get('balanced_accuracy', 0.0))
            macro_f1 = float(val_info.get('macro_f1', 0.0))
            auc_val = float(val_info.get('auc', 0.0))
            selection_score = 0.45 * bal_acc + 0.35 * macro_f1 + 0.20 * auc_val
            print(f"  SelectScore={selection_score:.4f}  "
                  f"(0.45×BalAcc + 0.35×MacroF1 + 0.20×AUC)")

            improved = (
                selection_score > self.best_selection_score + 1e-12
                or (
                    abs(selection_score - self.best_selection_score) <= 1e-12
                    and val_info['auc'] > self.best_val_auc
                )
            )

            if improved:
                self.best_selection_score = selection_score
                self.best_val_auc = val_info['auc']
                self.best_val_acc = val_info['accuracy']
                self.best_val_f1  = val_info['f1']
                self.best_val_bal_acc = float(val_info.get('balanced_accuracy', 0.0))
                self.best_val_macro_f1 = float(val_info.get('macro_f1', 0.0))
                self.best_threshold = float(val_info.get('threshold', 0.5))
                best_metrics = {k: v for k, v in val_info.items()}
                torch.save(self.model.state_dict(), save_path)

                # Save threshold alongside the checkpoint.
                try:
                    with open(save_path + '.threshold.json', 'w', encoding='utf-8') as f:
                        json.dump({
                            'threshold': self.best_threshold,
                            'selection_metric': '0.45*val_balanced_accuracy + 0.35*val_macro_f1 + 0.20*val_auc',
                            'selection_score': float(selection_score),
                            'val_accuracy': self.best_val_acc,
                            'val_balanced_accuracy': self.best_val_bal_acc,
                            'val_macro_f1': self.best_val_macro_f1,
                            'val_f1': self.best_val_f1,
                            'val_auc': self.best_val_auc,
                        }, f, indent=2, ensure_ascii=False)
                except Exception as exc:
                    print(f"  Threshold file save failed: {exc}")

                print(f"  Saved best model (Score: {self.best_selection_score:.4f}, "
                      f"BalAcc: {self.best_val_bal_acc:.4f}, MacroF1: {self.best_val_macro_f1:.4f}, "
                      f"ACC: {self.best_val_acc:.4f}, F1: {self.best_val_f1:.4f}, "
                      f"AUC: {self.best_val_auc:.4f}, Thr: {self.best_threshold:.4f})")
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= patience:
                print(f"\nEarly stopping ({patience} epochs without improvement)")
                break

        print("\n" + "=" * 70)
        print(f"Phase 3 done. Best Score={self.best_selection_score:.4f}  "
              f"BalAcc={self.best_val_bal_acc:.4f}  MacroF1={self.best_val_macro_f1:.4f}  "
              f"ACC={self.best_val_acc:.4f}  AUC={self.best_val_auc:.4f}  "
              f"F1={self.best_val_f1:.4f}  Thr={self.best_threshold:.4f}")
        print("=" * 70)

        return best_metrics

    def _train_epoch_phase3(
        self,
        loader,
        epoch: int,
        collusion_data: Optional[Dict],
    ) -> Dict[str, float]:
        """Phase 3 single-epoch training."""
        self.model.train()
        total_loss  = 0.0
        num_batches = 0
        start_time  = time.time()

        pbar = tqdm(loader, desc='Phase3-Train', leave=False)

        for batch_idx, batch_item in enumerate(pbar):
            if isinstance(batch_item, (tuple, list)):
                batch_data, time_mappings = batch_item[0], batch_item[1]
            else:
                batch_data, time_mappings = batch_item, None

            batch_data = batch_data.to(self.device, non_blocking=True)
            y = batch_data.y.long()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    p_hat, p_user, p_hat_0 = self._forward_phase3(
                        batch_data, time_mappings, collusion_data
                    )

                # Compute loss outside autocast.
                loss, loss_detail = self.compute_total_loss(
                    p_hat, p_user, p_hat_0, y
                )
            else:
                p_hat, p_user, p_hat_0 = self._forward_phase3(
                    batch_data, time_mappings, collusion_data
                )
                loss, loss_detail = self.compute_total_loss(
                    p_hat, p_user, p_hat_0, y
                )

            self._step_optimizer(loss, batch_idx)

            total_loss  += loss.item()
            num_batches += 1

            detail_str = {k: f'{v.item():.4f}' for k, v in loss_detail.items()}
            pbar.set_postfix({'loss': f'{loss.item():.4f}', **detail_str})

            if (batch_idx + 1) % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        n_samples = len(loader.dataset) if hasattr(loader, 'dataset') else num_batches

        return {
            'loss':       total_loss / max(num_batches, 1),
            'epoch_time': elapsed,
            'throughput': n_samples / elapsed,
        }

    def _forward_phase3(
        self,
        batch_data,
        time_mappings,
        collusion_data: Optional[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call model.forward() and handle legacy interfaces."""
        if hasattr(self.model, 'forward') and collusion_data is not None:
            try:
                out = self.model(batch_data, time_mappings,
                                 collusion_data=collusion_data)
            except TypeError:
                # Legacy interface without collusion_data
                out = self.model(batch_data, time_mappings)
        else:
            out = self.model(batch_data, time_mappings)

        if isinstance(out, (tuple, list)) and len(out) == 3:
            return out[0], out[1], out[2]
        elif isinstance(out, (tuple, list)) and len(out) == 2:
            # Legacy: (logits, aux), no p_hat_0
            logits = out[0]
            p_hat = torch.sigmoid(logits[:, 1] - logits[:, 0]) \
                    if logits.dim() == 2 else torch.sigmoid(logits)
            return p_hat, p_hat, p_hat
        else:
            raise RuntimeError(
                f"model.forward() returned {type(out)}; expected (p_hat, p_user, p_hat_0)."
            )

    # Evaluation

    @staticmethod
    def _metrics_at_threshold(labels: List[int], probs: List[float], threshold: float) -> Dict[str, float]:
        """Compute binary metrics at a threshold, incl. balanced accuracy and macro-F1."""
        preds = [int(p > threshold) for p in probs]
        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        return {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'macro_f1': float(macro_f1),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'threshold': float(threshold),
        }

    @staticmethod
    def _find_best_threshold(labels: List[int], probs: List[float]) -> Tuple[float, Dict[str, float]]:
        """Search for the best threshold on validation data."""
        labels_np = np.asarray(labels, dtype=int)
        probs_np = np.asarray(probs, dtype=float)

        if len(np.unique(labels_np)) < 2:
            return 0.5, MRDETGMTrainer._metrics_at_threshold(labels, probs, 0.5)

        # Candidate thresholds: unique probs + grid + 0.5
        eps = 1e-8
        uniq = np.unique(probs_np)
        candidates = np.concatenate([
            np.linspace(0.01, 0.99, 99),
            uniq - eps,
            uniq,
            uniq + eps,
            np.array([0.5]),
        ])
        candidates = np.unique(np.clip(candidates, 0.0, 1.0))

        best_t = 0.5
        best_m = MRDETGMTrainer._metrics_at_threshold(labels, probs, 0.5)
        best_tuple = (best_m['balanced_accuracy'], best_m['macro_f1'], best_m['f1'], best_m['accuracy'])

        for t in candidates:
            m = MRDETGMTrainer._metrics_at_threshold(labels, probs, float(t))
            key = (m['balanced_accuracy'], m['macro_f1'], m['f1'], m['accuracy'])
            if key > best_tuple:
                best_tuple = key
                best_t = float(t)
                best_m = m

        return best_t, best_m

    @torch.no_grad()
    def evaluate(
        self,
        loader,
        phase: int = 1,
        collusion_data: Optional[Dict] = None,
        threshold: Optional[float] = None,
        optimize_threshold: bool = False,
    ) -> Dict[str, float]:
        """Evaluate Phase 1 or Phase 3 models and return metrics."""
        self.model.eval()

        total_loss  = 0.0
        all_labels: List[int]   = []
        all_probs:  List[float] = []
        num_batches = 0
        eps = 1e-7

        for batch_item in loader:
            if isinstance(batch_item, (tuple, list)):
                batch_data, time_mappings = batch_item[0], batch_item[1]
            else:
                batch_data, time_mappings = batch_item, None

            batch_data = batch_data.to(self.device, non_blocking=True)
            y = batch_data.y.long()

            if phase == 1:
                _, _, _, p_user, p_hat_0 = self._forward_phase1(
                    batch_data, time_mappings
                )
                p_hat = p_hat_0
            else:
                p_hat, p_user, p_hat_0 = self._forward_phase3(
                    batch_data, time_mappings, collusion_data
                )

            # Loss uses probabilities and is independent of the threshold.
            p_hat_clamp = p_hat.clamp(eps, 1 - eps)
            loss = F.binary_cross_entropy(p_hat_clamp, y.float())
            total_loss += loss.item()
            num_batches += 1

            all_probs.extend(p_hat.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        if optimize_threshold:
            used_threshold, cls_metrics = self._find_best_threshold(all_labels, all_probs)
        else:
            if threshold is None:
                used_threshold = float(self.best_threshold) if phase == 3 else 0.5
            else:
                used_threshold = float(threshold)
            cls_metrics = self._metrics_at_threshold(all_labels, all_probs, used_threshold)

        preds = [int(p > used_threshold) for p in all_probs]
        if len(all_labels) > 0:
            tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
        else:
            tn = fp = fn = tp = 0

        return {
            'loss':      total_loss / max(num_batches, 1),
            'accuracy':  cls_metrics['accuracy'],
            'balanced_accuracy': cls_metrics.get('balanced_accuracy', 0.0),
            'macro_f1': cls_metrics.get('macro_f1', 0.0),
            'precision': cls_metrics['precision'],
            'recall':    cls_metrics['recall'],
            'f1':        cls_metrics['f1'],
            'auc':       float(auc),
            'threshold': float(used_threshold),
            'prob_min':  float(np.min(all_probs)) if all_probs else 0.0,
            'prob_max':  float(np.max(all_probs)) if all_probs else 0.0,
            'prob_mean': float(np.mean(all_probs)) if all_probs else 0.0,
            'pred_pos_rate': float(np.mean(preds)) if preds else 0.0,
            'n_samples': int(len(all_labels)),
            'n_real': int(sum(1 for y in all_labels if int(y) == 0)),
            'n_fake': int(sum(1 for y in all_labels if int(y) == 1)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        }

    # All-in-one training (three phases)

    def train_all_phases(
        self,
        train_loader,
        val_loader,
        twitter_aligner=None,
        full_dataset=None,
        phase1_save: str = 'checkpoints/phase1_best.pth',
        phase3_save: str = 'checkpoints/phase3_best.pth',
    ) -> Dict[str, Dict]:
        """Convenience method to run Phase 1 → Phase 2 → Phase 3."""
        # ── Phase 1 ────────────────────────────────────────────────────────
        p1_epochs = _cfg(self.config, 'PHASE1_EPOCHS', 30)
        p1_metrics = self.train_phase1(
            train_loader, val_loader,
            num_epochs=p1_epochs,
            save_path=phase1_save,
        )

        # Load best Phase 1 weights for Phase 2 graph build.
        self.model.load_state_dict(torch.load(phase1_save, map_location=self.device))
        print(f"\nLoaded Phase 1 best weights (AUC={p1_metrics.get('auc', 0):.4f})")

        # ── Phase 2 ────────────────────────────────────────────────────────
        collusion_data = self.build_phase2_graphs(
            full_dataset=full_dataset,
            twitter_aligner=twitter_aligner,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # ── Inject Phase 2 modules (required for Phase 3) ──────────────────
        twitter_id_map = twitter_aligner.user2idx if twitter_aligner is not None else {}
        base_model = getattr(self.model, '_orig_mod', self.model)
        base_model.initialize_phase2_modules(
            self.config, self.device, twitter_id_to_global_idx=twitter_id_map
        )

        # ── Phase 3 ────────────────────────────────────────────────────────
        p3_epochs = _cfg(self.config, 'PHASE3_EPOCHS', 50)
        p3_metrics = self.train_phase3(
            train_loader, val_loader,
            num_epochs=p3_epochs,
            collusion_data=collusion_data,
            save_path=phase3_save,
        )

        return {'phase1': p1_metrics, 'phase3': p3_metrics}

    # Legacy compatibility (OptimizedTrainer interface)

    def train_epoch(self, train_loader):
        """Backward compatibility: equivalent to a Phase 1 epoch."""
        return self._train_epoch_phase1(train_loader, 0)

    def train(self, train_loader, val_loader, num_epochs,
              save_path='best_model.pth'):
        """Backward compatibility: equivalent to train_phase1."""
        return self.train_phase1(
            train_loader, val_loader, num_epochs, save_path
        )


# Backward-compatible aliases

class OptimizedTrainer(MRDETGMTrainer):
    """Backward-compatible alias for legacy code."""
    pass


# Other aliases
ImprovedTrainer = MRDETGMTrainer
Trainer         = MRDETGMTrainer
