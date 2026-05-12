"""MIL and auxiliary losses for MR-DE-TGM."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def mil_loss(
    r_u_batch: List[torch.Tensor],
    pi_u_batch: List[torch.Tensor],
    y_batch: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run mil_loss."""
    assert len(r_u_batch) == len(pi_u_batch) == y_batch.shape[0], \
        "r_u_batch, pi_u_batch, and y_batch must have the same length"

    p_user_list = []

    for r_u, pi_u in zip(r_u_batch, pi_u_batch):

        r_u  = r_u.clamp(eps, 1.0 - eps)
        pi_u = pi_u.clamp(0.0, 1.0)


        log_complement = torch.log(1.0 - pi_u * r_u + eps)   # [N_i]
        log_complement_sum = log_complement.sum()              # scalar

        # p_i^user = 1 - exp(Σ log(1 - π×r))
        p_i = 1.0 - torch.exp(log_complement_sum)
        p_i = p_i.clamp(eps, 1.0 - eps)
        p_user_list.append(p_i)

    p_user = torch.stack(p_user_list)          # [B]
    loss   = F.binary_cross_entropy(p_user, y_batch.float())

    return loss, p_user


def mil_loss_batched(
    z_u: torch.Tensor,
    batch: torch.Tensor,
    risk_head: torch.nn.Module,
    attn_head: torch.nn.Module,
    y_batch: torch.Tensor,
    node_depth: Optional[torch.Tensor] = None,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run mil_loss_batched."""
    B = y_batch.shape[0]


    r_u = torch.sigmoid(risk_head(z_u)).squeeze(-1)            # [N]


    if node_depth is not None:
        pos = torch.stack([
            node_depth.float() / 10.0,
            (node_depth > 0).float()
        ], dim=-1)
        attn_input = torch.cat([z_u, pos], dim=-1)
    else:
        pad = torch.zeros(z_u.shape[0], 2, device=z_u.device)
        attn_input = torch.cat([z_u, pad], dim=-1)

    attn_raw = attn_head(attn_input).squeeze(-1)               # [N]


    pi_u = _graph_softmax(attn_raw, batch, B)                  # [N]


    p_user_list = []
    for i in range(B):
        mask = (batch == i)
        r_i  = r_u[mask].clamp(eps, 1.0 - eps)
        pi_i = pi_u[mask].clamp(0.0, 1.0)

        log_comp = torch.log(1.0 - pi_i * r_i + eps).sum()
        p_i = (1.0 - torch.exp(log_comp)).clamp(eps, 1.0 - eps)
        p_user_list.append(p_i)

    p_user = torch.stack(p_user_list)                          # [B]
    loss   = F.binary_cross_entropy(p_user, y_batch.float())

    return loss, p_user, r_u


def _graph_softmax(
    scores: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
) -> torch.Tensor:
    """Run _graph_softmax."""

    max_scores = torch.zeros(n_graphs, device=scores.device)
    max_scores.scatter_reduce_(0, batch, scores, reduce='amax', include_self=True)
    scores_shifted = scores - max_scores[batch]

    # exp
    exp_scores = torch.exp(scores_shifted)


    denom = torch.zeros(n_graphs, device=scores.device)
    denom.scatter_add_(0, batch, exp_scores)

    return exp_scores / (denom[batch] + 1e-9)


def consistency_loss(
    p_hat: torch.Tensor,
    p_group: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Run consistency_loss."""
    p     = p_hat.clamp(eps, 1.0 - eps)
    q     = p_group.clamp(eps, 1.0 - eps)

    # KL(Bern(p) || Bern(q)) = p*log(p/q) + (1-p)*log((1-p)/(1-q))
    kl = p * (torch.log(p) - torch.log(q)) + \
         (1.0 - p) * (torch.log(1.0 - p) - torch.log(1.0 - q))

    return kl.mean()


def symmetric_consistency_loss(
    p_hat: torch.Tensor,
    p_group: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Run symmetric_consistency_loss."""
    p = p_hat.clamp(eps, 1.0 - eps)
    q = p_group.clamp(eps, 1.0 - eps)
    m = 0.5 * (p + q)

    jsd = 0.5 * _bernoulli_kl(p, m, eps) + 0.5 * _bernoulli_kl(q, m, eps)
    return jsd.mean()


def _bernoulli_kl(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Run _bernoulli_kl."""
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))


def sparse_regularization(
    M_U: Optional[torch.Tensor] = None,
    lambda_U: float = 1e-4,
) -> torch.Tensor:
    """L1 sparsity penalty for the user collusion edge mask."""
    if M_U is None:
        return torch.tensor(0.0, requires_grad=True)
    return lambda_U * M_U.abs().mean()


def edge_mask_regularization(
    mask: torch.Tensor,
    target_density: float = 0.5,
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """Run edge_mask_regularization."""
    return lambda_reg * (mask.mean() - target_density).pow(2)


def balance_loss(S: torch.Tensor) -> torch.Tensor:
    """Run balance_loss."""
    assert S.dim() == 2, "S must be a 2D tensor with shape [U, K]"
    U, K = S.shape


    mean_assignment = S.mean(dim=0)                   # [K]
    target = torch.full_like(mean_assignment, 1.0 / K)

    loss = (mean_assignment - target).pow(2).sum()
    return loss


def entropy_regularization(S: torch.Tensor) -> torch.Tensor:
    """Run entropy_regularization."""
    eps = 1e-9
    S_clamped = S.clamp(eps, 1.0 - eps)
    entropy = -(S_clamped * torch.log(S_clamped)).sum(dim=-1).mean()
    return -entropy


class MRDETGMLossAggregator:
    """MRDETGMLossAggregator module."""

    def __init__(
        self,
        lambda_0: float = 0.5,   # L_init
        lambda_2: float = 0.3,   # L_MIL
        lambda_3: float = 0.2,   # L_cons
        lambda_4: float = 1e-4,  # L_sparse
        lambda_5: float = 0.1,   # L_balance
    ):
        self.lambda_0 = lambda_0
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.lambda_5 = lambda_5

    def compute(
        self,
        p_hat: torch.Tensor,
        y: torch.Tensor,
        p_hat_0: Optional[torch.Tensor] = None,
        p_user: Optional[torch.Tensor] = None,
        p_group: Optional[torch.Tensor] = None,
        M_U: Optional[torch.Tensor] = None,
        S: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Run compute."""
        y_f = y.float()
        details = {}


        L_event = F.binary_cross_entropy(
            p_hat.clamp(1e-7, 1 - 1e-7), y_f
        )
        details['L_event'] = L_event.item()
        total = L_event


        if p_hat_0 is not None:
            L_init = F.binary_cross_entropy(
                p_hat_0.clamp(1e-7, 1 - 1e-7), y_f
            )
            details['L_init'] = L_init.item()
            total = total + self.lambda_0 * L_init


        if p_user is not None:
            L_mil = F.binary_cross_entropy(
                p_user.clamp(1e-7, 1 - 1e-7), y_f
            )
            details['L_MIL'] = L_mil.item()
            total = total + self.lambda_2 * L_mil


        if p_group is not None and p_hat is not None:
            L_cons = consistency_loss(p_hat.detach(), p_group)
            details['L_cons'] = L_cons.item()
            total = total + self.lambda_3 * L_cons


        if M_U is not None:
            L_sparse = sparse_regularization(M_U, self.lambda_4)
            details['L_sparse'] = L_sparse.item()
            total = total + L_sparse


        if S is not None:
            L_bal = balance_loss(S)
            details['L_balance'] = L_bal.item()
            total = total + self.lambda_5 * L_bal

        details['total'] = total.item()
        return total, details
