import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Set, Optional


def calibration_loss(
    z: torch.Tensor, 
    centers: torch.Tensor, 
    assign: torch.Tensor, 
    temperature: float,
    R: Optional[Set[int]] = None,
) -> torch.Tensor:
    """
    Ranking-based calibration loss from MARK paper (Eq. 9).
    
    Brings uncertain nodes closer to their assigned cluster center
    while pushing them away from other clusters.
    
    L_cal = -1/|R| * sum_{i in R} log(exp(sim(h_i, mu_F)/t) / sum_k exp(sim(h_i, mu_k)/t))
    
    Args:
        z: Node embeddings [N, D]
        centers: Cluster centers [K, D]
        assign: Cluster assignments for each node [N]
        temperature: Temperature for softmax scaling
        R: Optional set of reliable node indices to use. If None, uses all nodes.
    
    Returns:
        Calibration loss scalar
    """
    # Normalize embeddings and centers
    z_n = F.normalize(z, p=2, dim=-1)
    c_n = F.normalize(centers, p=2, dim=-1)
    
    # If R is provided, only compute loss for those nodes
    if R is not None and len(R) > 0:
        R_list = sorted(list(R))
        z_n = z_n[R_list]
        assign = assign[R_list]
    
    if z_n.size(0) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    
    # Compute similarity logits: [N, K]
    logits = torch.matmul(z_n, c_n.t()) / temperature
    
    # Cross-entropy loss with assigned clusters as targets
    loss = F.cross_entropy(logits, assign)
    
    return loss


def ranking_calibration_loss(
    z: torch.Tensor,
    centers: torch.Tensor,
    pred_labels: Dict[int, int],
    temperature: float,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Enhanced ranking-based calibration with margin.
    
    Uses LLM predictions as soft targets and includes a margin term
    to ensure confident separation between correct and incorrect clusters.
    
    Args:
        z: Node embeddings [N, D]
        centers: Cluster centers [K, D]
        pred_labels: Dict mapping node_id -> predicted_cluster_id from LLM
        temperature: Temperature scaling
        margin: Margin for contrastive loss
    
    Returns:
        Ranking loss scalar
    """
    if not pred_labels:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    
    z_n = F.normalize(z, p=2, dim=-1)
    c_n = F.normalize(centers, p=2, dim=-1)
    
    node_ids = list(pred_labels.keys())
    labels = torch.tensor([pred_labels[n] for n in node_ids], device=z.device)
    
    # Get embeddings for labeled nodes
    z_labeled = z_n[node_ids]
    
    # Compute similarities
    sims = torch.matmul(z_labeled, c_n.t())  # [|R|, K]
    
    # Get positive similarities (to assigned cluster)
    pos_sims = sims[torch.arange(len(node_ids), device=z.device), labels]
    
    # Contrastive loss with margin
    # For each node, we want sim(node, assigned_center) > sim(node, other_centers) + margin
    K = centers.size(0)
    mask = torch.ones_like(sims, dtype=torch.bool)
    mask[torch.arange(len(node_ids), device=z.device), labels] = False
    
    # Max negative similarity
    neg_sims = sims.masked_fill(~mask, float('-inf')).max(dim=1).values
    
    # Hinge loss: max(0, margin - (pos - neg))
    losses = F.relu(margin - (pos_sims - neg_sims))
    
    return losses.mean()


def rank_finetune_loss(
    L_eng: torch.Tensor, 
    z: torch.Tensor, 
    centers: torch.Tensor, 
    assign: torch.Tensor, 
    temperature: float, 
    weight: float,
    R: Optional[Set[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined fine-tuning loss: L_ft = L_eng + weight * L_cal
    
    Args:
        L_eng: Engine loss (contrastive + clustering)
        z: Node embeddings
        centers: Cluster centers
        assign: Cluster assignments
        temperature: Temperature for calibration loss
        weight: Weight for calibration loss
        R: Set of reliable nodes for calibration
    
    Returns:
        Total loss and component dict
    """
    l_cal = calibration_loss(z, centers, assign, temperature, R)
    total = L_eng + weight * l_cal
    return total, {"Lcal": float(l_cal.item())}


def compute_uncertainty_scores(
    z: torch.Tensor,
    centers: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute uncertainty scores for each node based on cluster assignment confidence.
    Higher score = more uncertain.
    
    Args:
        z: Node embeddings [N, D]
        centers: Cluster centers [K, D]
        temperature: Temperature for softmax
    
    Returns:
        Uncertainty scores [N]
    """
    z_n = F.normalize(z, p=2, dim=-1)
    c_n = F.normalize(centers, p=2, dim=-1)
    
    # Compute soft assignments
    logits = torch.matmul(z_n, c_n.t()) / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Entropy as uncertainty measure
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    return entropy
