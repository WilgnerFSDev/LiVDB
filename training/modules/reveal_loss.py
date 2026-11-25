import torch
import torch.nn as nn
import torch.nn.functional as F


class ReVealLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.001,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        hg: torch.Tensor,
        yg: torch.Tensor,
        lg: torch.Tensor,
        hp: torch.Tensor,
        hn: torch.Tensor,
    ) -> torch.Tensor:
        Lce = F.binary_cross_entropy(yg, lg.float(), reduction="sum")

        D_same = 1.0 - F.cosine_similarity(hg, hp)
        D_diff = 1.0 - F.cosine_similarity(hg, hn)
        Lp = torch.abs(D_same - D_diff + self.gamma).sum()

        Lreg = (hg.norm(dim=1) + hp.norm(dim=1) + hn.norm(dim=1)).sum()

        return Lce + self.alpha * Lp + self.beta * Lreg
