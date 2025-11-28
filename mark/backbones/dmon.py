from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DMoNEncoder(nn.Module):
    """
    DMoN: Deep Modularity Networks for Graph Clustering.
    
    Based on Tsitsulin et al. 2023 - optimizes modularity directly.
    Uses a GNN encoder followed by a soft assignment head.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_clusters: int,
        num_layers: int = 2,
        collapse_regularization: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.collapse_reg = collapse_regularization
        self.num_clusters = num_clusters
        self.dropout = dropout
        
        # Multi-layer GCN encoder
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Soft assignment head
        self.assignment = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters),
        )

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GCN encoding."""
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h: node embeddings [N, hidden_dim]
            S: soft cluster assignments [N, K]
        """
        h = self._encode(x, edge_index)
        logits = self.assignment(h)
        S = F.softmax(logits, dim=-1)
        return h, S

    def _compute_modularity(
        self, 
        S: torch.Tensor, 
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute modularity loss.
        Q = 1/(2m) * Tr(S^T * B * S)
        where B = A - d*d^T/(2m) is the modularity matrix.
        """
        # Build adjacency matrix
        row, col = edge_index
        values = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Compute A @ S efficiently using sparse-dense multiplication
        # A is symmetric, so we count both directions
        A_S = torch.zeros(num_nodes, self.num_clusters, device=S.device)
        A_S.index_add_(0, row, S[col])
        
        # Degree vector
        degrees = torch.zeros(num_nodes, device=S.device)
        degrees.index_add_(0, row, values)
        
        # Total edges (each edge counted twice in undirected graph)
        m = degrees.sum() / 2 + 1e-8
        
        # Compute S^T @ A @ S
        S_A_S = torch.matmul(S.t(), A_S)  # [K, K]
        
        # Compute S^T @ d and d^T @ S for the degree correction term
        S_d = torch.matmul(S.t(), degrees.unsqueeze(1))  # [K, 1]
        degree_term = torch.matmul(S_d, S_d.t()) / (2 * m)  # [K, K]
        
        # Modularity: Tr(S^T @ B @ S) / (2m) = Tr(S^T @ A @ S - degree_term) / (2m)
        modularity = (torch.trace(S_A_S) - torch.trace(degree_term)) / (2 * m)
        
        return modularity

    def _collapse_loss(self, S: torch.Tensor) -> torch.Tensor:
        """
        Collapse regularization to prevent all nodes going to one cluster.
        Encourages uniform cluster sizes.
        """
        # Average assignment per cluster
        cluster_sizes = S.mean(dim=0)  # [K]
        # Entropy of cluster distribution - maximize it for uniform sizes
        entropy = -(cluster_sizes * torch.log(cluster_sizes + 1e-8)).sum()
        # We want to maximize entropy, so return negative
        return -entropy

    def loss_pretrain(
        self, 
        S: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        DMoN loss: maximize modularity + collapse regularization.
        """
        num_nodes = S.size(0)
        
        modularity = self._compute_modularity(S, edge_index, num_nodes)
        collapse = self._collapse_loss(S)
        
        # Maximize modularity = minimize -modularity
        loss = -modularity + self.collapse_reg * collapse
        
        return loss, {
            "Modularity": float(modularity.item()), 
            "Collapse": float(collapse.item())
        }

    def predict_assign(self, outputs) -> torch.Tensor:
        """Get hard cluster assignments."""
        if isinstance(outputs, tuple):
            _, S = outputs
        else:
            S = outputs
        return S.argmax(dim=1)
    
    def soft_assign(self, outputs) -> torch.Tensor:
        """Get soft cluster assignments."""
        if isinstance(outputs, tuple):
            _, S = outputs
        else:
            S = outputs
        return S
