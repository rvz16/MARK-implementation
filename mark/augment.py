import torch


def feature_dropout(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    if p <= 0:
        return x
    mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
    return x * mask


def edge_dropout(edge_index: torch.Tensor, drop: float = 0.2) -> torch.Tensor:
    if drop <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop
    kept = edge_index[:, keep_mask]
    return kept
