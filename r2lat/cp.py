"""CP and FF sampling kernels for the R2 latency microbenchmark.

We use random weights. The benchmark measures the computational cost of the
sampling path; numerical values are irrelevant. Both samplers expose a
`.sample(h)` method for uniformity: input (n, H) drafter features, output
(n,) token ids.

SharedTrunkCPSampler
--------------------
Mirrors `factorization_gap/fgap/heads.py::SharedTrunkCPHead` but exposes
sampling rather than NLL. Rank-r CP joint with shared LM-head trunk
W_LM ∈ (V, H). For rank component α:

    logits_α(i) = W_LM @ (w_sh[i, α] ⊙ h_i)
    log p_α(x|h) = sum_i log softmax(logits_α(i))[x_i]
    p(x|h) = sum_α π_α(h) · p_α(x|h)

Sampling is sequential *within* the block (Basharin 2024 §3.4, eq. 13):
after sampling positions 0..j-1 the posterior over mixture components
updates as π_α^{(j)} ∝ π_α · prod_{i<j} p_α(x_i | h_i), which we maintain
in log-space via logsumexp.

The precompute matmul is O(n·r·V·H), dominated by reading W_LM (V·H bf16)
from HBM. Sequential sampling adds n small kernels; that overhead is what
R2 is measuring against the target-forward denominator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTrunkCPSampler(nn.Module):
    """Rank-r shared-trunk CP joint sampler over one block of size n."""

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        rank: int,
        vocab_size: int,
        lm_head_weight: torch.Tensor,  # (V, H), shared across samplers
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert lm_head_weight.shape == (vocab_size, hidden_size), (
            f"lm_head_weight shape {tuple(lm_head_weight.shape)} "
            f"!= ({vocab_size}, {hidden_size})"
        )
        self.n = block_size
        self.r = rank
        self.V = vocab_size
        self.H = hidden_size
        # Reference to shared trunk — not a parameter of this module.
        self.register_buffer("W_LM", lm_head_weight, persistent=False)
        # Init near 1 so dummy outputs are well-scaled.
        w = torch.ones(block_size, rank, hidden_size, device=device, dtype=dtype)
        w = w + 0.02 * torch.randn_like(w)
        self.w_sh = nn.Parameter(w)
        self.gate = nn.Linear(hidden_size, rank, device=device, dtype=dtype)
        self.eval()

    @torch.inference_mode()
    def sample(self, h: torch.Tensor) -> torch.Tensor:
        """h: (n, H) — drafter features for one fully-masked block."""
        n, r, V = self.n, self.r, self.V

        # Mixture log-prior from mean-pooled h.
        pool = h.mean(dim=0)  # (H,)
        log_pi = F.log_softmax(self.gate(pool), dim=-1).float()  # (r,)

        # Per-position per-rank modulated features then shared lm_head.
        h_exp = self.w_sh * h.unsqueeze(1)  # (n, r, H)
        logits = F.linear(h_exp.reshape(n * r, self.H), self.W_LM)  # (n*r, V)
        logits = logits.float().reshape(n, r, V)
        log_p = F.log_softmax(logits, dim=-1)  # (n, r, V)

        # Sequential (within-block) sampling with logsumexp mixture updates.
        x = torch.empty(n, dtype=torch.long, device=h.device)
        for j in range(n):
            # p(x_j | x_<j, h) = logsumexp_α ( log π_α^{(j)} + log p_α(x_j | h_j) )
            log_joint = torch.logsumexp(
                log_pi.unsqueeze(-1) + log_p[j], dim=0
            )  # (V,)
            sampled = torch.multinomial(log_joint.exp(), num_samples=1)  # (1,)
            x[j] = sampled
            # Posterior over mixture components given the new sample.
            log_term = log_pi + log_p[j].index_select(-1, sampled).squeeze(-1)
            log_pi = log_term - torch.logsumexp(log_term, dim=0)
        return x


class FFSampler(nn.Module):
    """Factorized single-head sampler — the deployed DFlash baseline.

    Single W_LM matmul → per-position softmax → per-position multinomial.
    """

    def __init__(
        self,
        hidden_size: int,
        block_size: int,
        vocab_size: int,
        lm_head_weight: torch.Tensor,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        assert lm_head_weight.shape == (vocab_size, hidden_size)
        self.n = block_size
        self.V = vocab_size
        self.H = hidden_size
        self.register_buffer("W_LM", lm_head_weight, persistent=False)
        self.eval()

    @torch.inference_mode()
    def sample(self, h: torch.Tensor) -> torch.Tensor:
        logits = F.linear(h, self.W_LM)  # (n, V)
        probs = F.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)  # (n,)
