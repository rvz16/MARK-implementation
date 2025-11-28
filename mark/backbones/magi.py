from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MAGIEncoder(nn.Module):
    """
    MAGI: Multi-view Alignment Graph Clustering with Instance-level supervision.
    
    Based on the paper's description:
    - Uses dual-view contrastive learning with NT-Xent loss
    - Includes clustering loss for cluster center alignment
    - Supports proper center initialization with K-means
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        proj_dim: int,
        num_clusters: int,
        tau_align: float = 0.5,
        lambda_clu: float = 1.0,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tau_align = tau_align
        self.lambda_clu = lambda_clu
        self.num_clusters = num_clusters
        self.proj_dim = proj_dim
        self.dropout = dropout
        
        # Multi-layer GCN encoder
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Projection head (2-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        # Cluster centers (learnable parameters)
        self._centers = nn.Parameter(torch.randn(num_clusters, proj_dim))
        self._centers_initialized = False
        
    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """GCN encoding with batch norm and activation."""
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized projections."""
        h = self._encode(x, edge_index)
        z = self.projector(h)
        z = F.normalize(z, p=2, dim=-1)
        return z
    
    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get hidden embeddings before projection."""
        return self._encode(x, edge_index)

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
        This is the contrastive loss for aligning two views.
        """
        N = z1.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.t()) / self.tau_align
        
        # Labels are the diagonal (positive pairs)
        labels = torch.arange(N, device=z1.device)
        
        # Cross entropy from both directions
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        
        return 0.5 * (loss_12 + loss_21)

    def _cluster_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Clustering loss that encourages nodes to be close to cluster centers.
        Uses soft assignment based on distance to centers.
        """
        centers = F.normalize(self._centers, p=2, dim=-1)
        
        # Compute distances to all centers
        # z: [N, D], centers: [K, D]
        sim = torch.matmul(z, centers.t())  # [N, K], higher is more similar
        
        # Soft assignment (Student's t-distribution kernel, like t-SNE/DEC)
        # q_ij = (1 + ||z_i - c_j||^2)^-1 / sum_k (1 + ||z_i - c_k||^2)^-1
        dist_sq = 2 - 2 * sim  # Since ||z - c||^2 = 2 - 2*cos(z,c) for normalized vectors
        q = 1.0 / (1.0 + dist_sq)
        q = q / q.sum(dim=1, keepdim=True)
        
        # Target distribution (sharpen the soft assignments)
        # p_ij = q_ij^2 / f_j / sum_k (q_ik^2 / f_k)
        f = q.sum(dim=0)  # frequency per cluster
        p = (q ** 2) / f.unsqueeze(0)
        p = p / p.sum(dim=1, keepdim=True)
        
        # KL divergence loss
        loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')
        
        return loss

    def _simple_cluster_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Simple clustering loss - minimize distance to nearest center."""
        centers = F.normalize(self._centers, p=2, dim=-1)
        sim = torch.matmul(z, centers.t())  # [N, K]
        # Maximize similarity to nearest center = minimize -max_sim
        max_sim = sim.max(dim=1).values
        return -max_sim.mean()

    def loss_pretrain(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Pretraining loss combining contrastive alignment and clustering.
        L_eng = L_ali + lambda * L_clu
        """
        # Contrastive alignment loss
        l_align = self._nt_xent(z1, z2)
        
        # Clustering loss on both views
        l_clu = 0.5 * (self._cluster_loss(z1) + self._cluster_loss(z2))
        
        # Combined loss
        loss = l_align + self.lambda_clu * l_clu
        
        return loss, {"Lali": float(l_align.item()), "Lclu": float(l_clu.item())}

    def predict_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Predict cluster assignments based on nearest center."""
        centers = F.normalize(self._centers, p=2, dim=-1)
        sim = torch.matmul(z, centers.t())
        return sim.argmax(dim=1)
    
    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Get soft cluster assignments (probabilities)."""
        centers = F.normalize(self._centers, p=2, dim=-1)
        sim = torch.matmul(z, centers.t())
        return F.softmax(sim / self.tau_align, dim=-1)

    def centers(self) -> torch.Tensor:
        """Return normalized cluster centers."""
        return F.normalize(self._centers, p=2, dim=-1).detach()
    
    def init_centers_kmeans(self, z: torch.Tensor, max_iter: int = 100):
        """
        Initialize cluster centers using K-means on the embeddings.
        This should be called after initial pretraining.
        """
        from sklearn.cluster import KMeans
        
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=20, max_iter=max_iter, random_state=42)
        kmeans.fit(z_np)
        
        centers = torch.tensor(kmeans.cluster_centers_, dtype=z.dtype, device=z.device)
        centers = F.normalize(centers, p=2, dim=-1)
        
        with torch.no_grad():
            self._centers.copy_(centers)
        
        self._centers_initialized = True
        print(f"[MAGI] Initialized centers with K-means, inertia={kmeans.inertia_:.4f}")
        
        return kmeans.labels_
