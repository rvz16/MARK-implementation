/**
 * MARK Blog - Interactive Visualizations
 * 
 * This file contains all the interactive graph visualizations
 * and animations for the MARK framework blog.
 */

// ============================================
// Configuration - UPDATE THESE FOR YOUR REPO
// ============================================

const CONFIG = {
    // Replace with your actual GitHub username and repo
    GITHUB_REPO: 'https://github.com/Quartz-Admirer/MARK-implementation',
    // Replace with actual paper URL when available
    PAPER_URL: 'https://aclanthology.org/2025.findings-acl.314.pdf',
};

// ============================================
// File Contents for Code Viewer
// ============================================

const FILE_CONTENTS = {
    // mark/datasets.py
    datasets: `import os
import json
import urllib.request
from typing import List, Tuple, Optional

import torch
import numpy as np
from torch_geometric.datasets import Planetoid, WikiCS

# URLs for TAG benchmark datasets with raw texts
TAG_URLS = {
    "cora": "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/data/cora/raw_texts.json",
    "citeseer": "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/data/citeseer/raw_texts.json",
    "pubmed": "https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/data/pubmed/raw_texts.json",
}

# Class names for each dataset
CLASS_NAMES = {
    "cora": [
        "Case_Based", "Genetic_Algorithms", "Neural_Networks",
        "Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", "Theory"
    ],
    "citeseer": ["Agents", "AI", "DB", "IR", "ML", "HCI"],
    "pubmed": [
        "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2", 
        "Diabetes Mellitus Experimental"
    ],
}

def load_tag(name: str, root: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor, int]:
    """
    Load text-attributed graph datasets.
    Returns: X, edge_index, texts, y, K (number of classes)
    """
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    
    if name.lower() in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=root, name=name.capitalize())
    elif name.lower() == "wikics":
        dataset = WikiCS(root=root)
    else:
        raise ValueError(f"Unsupported dataset {name}")

    data = dataset[0]
    x = data.x.float()
    edge_index = data.edge_index
    y = data.y.long()
    K = int(y.max().item() + 1)
    
    # Try to load raw texts
    texts = _download_raw_texts(name.lower(), root)
    
    print(f"[DATA] Loaded {name} |V|={x.size(0)} |E|={edge_index.size(1)} K={K}")
    return x, edge_index, texts, y, K`,

    // mark/augment.py
    augment: `import torch


def feature_dropout(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """
    Randomly drop node features (augmentation for contrastive learning).
    
    Args:
        x: Node feature matrix [N, F]
        p: Dropout probability
    
    Returns:
        Augmented feature matrix with some features zeroed out
    """
    if p <= 0:
        return x
    mask = torch.bernoulli(torch.ones_like(x) * (1 - p))
    return x * mask


def edge_dropout(edge_index: torch.Tensor, drop: float = 0.2) -> torch.Tensor:
    """
    Randomly drop edges (augmentation for contrastive learning).
    
    Args:
        edge_index: Edge index tensor [2, E]
        drop: Dropout probability
    
    Returns:
        Augmented edge index with some edges removed
    """
    if drop <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop
    kept = edge_index[:, keep_mask]
    return kept`,

    // mark/metrics.py
    metrics: `import json
import csv
from typing import Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm.
    
    The Hungarian algorithm finds the optimal one-to-one mapping
    between predicted clusters and true labels.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = float(w[row_ind, col_ind].sum()) / y_pred.size
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    
    return acc, mapping


def compute_all(y_true, y_pred) -> Dict[str, float]:
    """
    Compute all clustering evaluation metrics:
    - ACC: Clustering accuracy (with Hungarian matching)
    - NMI: Normalized Mutual Information
    - ARI: Adjusted Rand Index
    - F1: Macro F1-score
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    acc, mapping = cluster_acc(y_true, y_pred)
    nmi = float(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
    ari = float(adjusted_rand_score(y_true, y_pred))
    
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    f1 = float(f1_score(y_true, y_pred_mapped, average="macro", zero_division=0))
    
    return {"ACC": acc, "NMI": nmi, "ARI": ari, "F1": f1}`,

    // mark/ranking.py
    ranking: `import torch
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
    
    L_cal = -1/|R| * sum_{i in R} log(
        exp(sim(h_i, mu_F)/t) / sum_k exp(sim(h_i, mu_k)/t)
    )
    """
    z_n = F.normalize(z, p=2, dim=-1)
    c_n = F.normalize(centers, p=2, dim=-1)
    
    # Filter to reliable nodes only
    if R is not None and len(R) > 0:
        R_list = sorted(list(R))
        z_n = z_n[R_list]
        assign = assign[R_list]
    
    if z_n.size(0) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    
    # Compute similarity logits
    logits = torch.matmul(z_n, c_n.t()) / temperature
    
    # Cross-entropy with assigned clusters as targets
    loss = F.cross_entropy(logits, assign)
    
    return loss


def compute_uncertainty_scores(
    z: torch.Tensor,
    centers: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute uncertainty scores based on cluster assignment entropy.
    Higher score = more uncertain.
    """
    z_n = F.normalize(z, p=2, dim=-1)
    c_n = F.normalize(centers, p=2, dim=-1)
    
    logits = torch.matmul(z_n, c_n.t()) / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Entropy as uncertainty
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return entropy`,

    // mark/backbones/magi.py
    magi: `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MAGIEncoder(nn.Module):
    """
    MAGI: Multi-view Alignment Graph Clustering with Instance-level supervision.
    
    Combines contrastive learning (NT-Xent) with clustering objectives.
    Uses dual-view augmentation for self-supervised learning.
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
        
        # Multi-layer GCN encoder
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        
        # Learnable cluster centers
        self._centers = nn.Parameter(torch.randn(num_clusters, proj_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self._encode(x, edge_index)
        z = self.projector(h)
        z = F.normalize(z, p=2, dim=-1)
        return z

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent contrastive loss."""
        N = z1.size(0)
        sim_matrix = torch.matmul(z1, z2.t()) / self.tau_align
        labels = torch.arange(N, device=z1.device)
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        return 0.5 * (loss_12 + loss_21)

    def loss_pretrain(self, z1, z2):
        """Combined loss: L = L_ali + λ * L_clu"""
        l_align = self._nt_xent(z1, z2)
        l_clu = 0.5 * (self._cluster_loss(z1) + self._cluster_loss(z2))
        loss = l_align + self.lambda_clu * l_clu
        return loss, {"Lali": l_align.item(), "Lclu": l_clu.item()}`,

    // mark/backbones/dmon.py
    dmon: `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DMoNEncoder(nn.Module):
    """
    DMoN: Deep Modularity Networks for Graph Clustering.
    
    Directly optimizes graph modularity as training objective.
    Uses soft cluster assignments and collapse regularization.
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
        
        # GCN encoder
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

    def forward(self, x, edge_index):
        h = self._encode(x, edge_index)
        logits = self.assignment(h)
        S = F.softmax(logits, dim=-1)  # Soft assignments
        return h, S

    def _compute_modularity(self, S, edge_index, num_nodes):
        """Differentiable modularity computation."""
        row, col = edge_index
        A_S = torch.zeros(num_nodes, self.num_clusters, device=S.device)
        A_S.index_add_(0, row, S[col])
        
        degrees = torch.zeros(num_nodes, device=S.device)
        degrees.index_add_(0, row, torch.ones(len(row), device=S.device))
        m = degrees.sum() / 2 + 1e-8
        
        S_A_S = torch.matmul(S.t(), A_S)
        S_d = torch.matmul(S.t(), degrees.unsqueeze(1))
        degree_term = torch.matmul(S_d, S_d.t()) / (2 * m)
        
        modularity = (torch.trace(S_A_S) - torch.trace(degree_term)) / (2 * m)
        return modularity

    def loss_pretrain(self, S, edge_index):
        """Loss = -Modularity + collapse_reg * Collapse"""
        modularity = self._compute_modularity(S, edge_index, S.size(0))
        collapse = self._collapse_loss(S)
        loss = -modularity + self.collapse_reg * collapse
        return loss, {"Modularity": modularity.item()}`,

    // mark/agents/concept.py
    concept: `import json
import os
from typing import Dict, List, Optional, Any

import torch

from mark.agents.client import LLMClient
from mark.agents.prompts import CONCEPT_PROMPT


def induce_concepts(
    H: torch.Tensor,
    assign: torch.Tensor,
    top_n: int,
    client: LLMClient,
    texts: Optional[List[str]] = None,
    run_dir: Optional[str] = None,
):
    """
    Concept Agent: Induce cluster concepts from top-n high-confidence samples.
    
    For each cluster:
    1. Find the cluster center
    2. Select top-N nodes closest to center (highest confidence)
    3. Send their texts to LLM for concept induction
    4. LLM returns cluster title and keywords
    
    Args:
        H: Node embeddings [N, D]
        assign: Cluster assignments [N]
        top_n: Number of top samples per cluster
        client: LLM client for API calls
        texts: Raw text descriptions for each node
    
    Returns:
        Dict mapping cluster_id -> {"title": str, "keywords": list}
    """
    texts = texts or [f"node_{i}" for i in range(H.size(0))]
    clusters = assign.unique().tolist()
    
    messages = []
    cluster_ids = []
    
    for k in clusters:
        idx = (assign == k).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        
        # Compute cluster center
        center = F.normalize(H[idx].mean(dim=0, keepdim=True), p=2, dim=-1)
        
        # Find top-N closest nodes
        h = F.normalize(H[idx], p=2, dim=-1)
        sims = (h @ center.t()).view(-1)
        top_idx = sims.topk(min(top_n, idx.numel())).indices
        selected = idx[top_idx].tolist()
        
        # Prepare prompt with selected texts
        phrases = [texts[j] for j in selected]
        user_content = f"Cluster {k} top nodes:\\n" + "\\n".join(phrases[:20])
        messages.append([
            {"role": "system", "content": CONCEPT_PROMPT},
            {"role": "user", "content": user_content}
        ])
        cluster_ids.append(k)
    
    # Batch call to LLM
    responses = client.batch_chat(messages)
    
    results = {}
    for k, (resp, usage) in zip(cluster_ids, responses):
        title = resp.get("cluster_title", "") or resp.get("title", "")
        keywords = resp.get("keywords", [])
        results[k] = {"title": title, "keywords": keywords}
    
    return results`,

    // mark/agents/generation.py
    generation: `import json
import os
from typing import Dict, Iterable, List, Optional, Any

from mark.agents.client import LLMClient
from mark.agents.prompts import GENERATION_PROMPT


def synthesize_for_S(
    S: Iterable[int],
    neighbors: Dict[int, List[int]],
    texts: List[str],
    concepts: Dict[int, Dict[str, Any]],
    k: int,
    client: LLMClient,
    run_dir: Optional[str] = None,
) -> Dict[int, str]:
    """
    Generation Agent: Synthesize virtual text for uncertain nodes.
    
    For each uncertain node in S:
    1. Gather K nearest neighbors' texts
    2. Include cluster concept information
    3. Ask LLM to synthesize a summary combining node + neighborhood
    
    This creates an augmented representation that incorporates
    graph structure information into the text.
    
    Args:
        S: Set of uncertain node indices
        neighbors: Adjacency list {node_id: [neighbor_ids]}
        texts: Raw text for each node
        concepts: Cluster concepts from Concept Agent
        k: Number of neighbors to include
        client: LLM client
    
    Returns:
        Dict mapping node_id -> synthesized summary text
    """
    S_list = list(S)
    if not S_list:
        return {}
    
    messages = []
    
    # Format cluster concepts for context
    concept_str = "\\n".join(
        f"Cluster {cid}: {info.get('title', '')} "
        f"(Keywords: {', '.join(info.get('keywords', []))})"
        for cid, info in concepts.items()
    )

    for i in S_list:
        neigh_ids = neighbors.get(i, [])[:k]
        neigh_texts = [texts[j] for j in neigh_ids if j < len(texts)]
        
        user_content = f\"\"\"Node: {texts[i] if i < len(texts) else f"node_{i}"}
Neighbors:
{chr(10).join(f"- {t}" for t in neigh_texts) if neigh_texts else "- none"}

Cluster Concepts:
{concept_str}\"\"\"
        
        messages.append([
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": user_content}
        ])

    responses = client.batch_chat(messages)
    
    results = {}
    for node_id, (resp, usage) in zip(S_list, responses):
        summary = resp.get("summary", "")
        results[node_id] = summary
    
    return results`,

    // mark/agents/inference.py
    inference: `import json
import os
from typing import Dict, Iterable, Set, Optional, Any

from mark.agents.client import LLMClient
from mark.agents.prompts import INFERENCE_PROMPT


def classify_consistency(
    S: Iterable[int],
    raw_texts: Dict[int, str],
    synth_texts: Dict[int, str],
    concepts: Dict[int, Dict[str, Any]],
    K: int,
    client: LLMClient,
    run_dir: Optional[str] = None,
):
    """
    Inference Agent: Classify nodes and filter by consistency.
    
    For each uncertain node:
    1. Classify based on ORIGINAL text -> pred_original
    2. Classify based on SYNTHETIC text -> pred_synthetic
    3. If pred_original == pred_synthetic: node is RELIABLE (add to R)
    
    This consistency check filters out nodes where the LLM
    is uncertain or where original/synthetic disagree.
    
    Args:
        S: Set of uncertain node indices
        raw_texts: Original texts for each node
        synth_texts: Synthesized texts from Generation Agent
        concepts: Cluster concepts for classification
        K: Number of clusters
        client: LLM client
    
    Returns:
        R: Set of reliable node indices
        stats: Statistics per cluster
        pred_labels: Dict of node_id -> predicted_cluster
    """
    S_list = list(S)
    if not S_list:
        return set(), {k: {"agree": 0, "total": 0} for k in range(K)}, {}
    
    # Format cluster concepts
    concept_str = "\\n".join(
        f"Cluster {cid}: {info.get('title', '')} "
        f"(Keywords: {', '.join(info.get('keywords', []))})"
        for cid, info in concepts.items()
    )

    # Prepare messages for both original and synthetic classification
    messages_raw = []
    messages_syn = []
    
    for i in S_list:
        orig = raw_texts.get(i, "")
        summ = synth_texts.get(i, "")
        
        user_raw = f"Original text:\\n{orig}\\n\\nCluster Concepts:\\n{concept_str}"
        user_syn = f"Synthetic text:\\n{summ}\\n\\nCluster Concepts:\\n{concept_str}"
        
        messages_raw.append([
            {"role": "system", "content": INFERENCE_PROMPT},
            {"role": "user", "content": user_raw}
        ])
        messages_syn.append([
            {"role": "system", "content": INFERENCE_PROMPT},
            {"role": "user", "content": user_syn}
        ])

    resp_raw = client.batch_chat(messages_raw)
    resp_syn = client.batch_chat(messages_syn)

    R: Set[int] = set()
    pred_labels: Dict[int, int] = {}
    
    for node_id, raw_pack, syn_pack in zip(S_list, resp_raw, resp_syn):
        cid_raw = int(raw_pack[0].get("cluster_id", -1))
        cid_syn = int(syn_pack[0].get("cluster_id", -1))
        
        # Consistency check: both predict same cluster
        if cid_raw == cid_syn and 0 <= cid_raw < K:
            R.add(node_id)
            pred_labels[node_id] = cid_raw
    
    return R, {}, pred_labels`,

    // mark/agents/prompts.py
    prompts: `# Concept Agent Prompt
CONCEPT_PROMPT = """You are an AI assistant specializing in text induction. 
Your task is to generate a topic name based on the provided input texts.

Analyze the commonalities and core content of the samples provided 
and identify the main theme or topic they share.

Your response must be ONLY a JSON object with the following structure:
{
    "cluster_title": "A short, descriptive topic name (2-5 words)",
    "keywords": ["keyword1", "keyword2", "keyword3"]
}

Be concise and precise. The topic name should capture the essence of all samples."""


# Generation Agent Prompt
GENERATION_PROMPT = """You are an AI assistant specializing in text synthesis.
Your task is to create a virtual summary text based on a target node 
and its neighbors in a graph.

Given information about a target article/node and its neighboring nodes, 
generate a concise summary that:
1. Captures the main topic of the target
2. Incorporates relevant information from neighbors
3. Maintains consistency with the cluster concepts provided

Your response must be ONLY a JSON object with the following structure:
{
    "summary": "A 2-3 sentence summary that synthesizes the target with neighborhood context"
}

Be factual and concise. The summary should be self-contained and informative."""


# Inference Agent Prompt  
INFERENCE_PROMPT = """You are an AI assistant specializing in text classification.
Your task is to identify the most likely cluster to which a given text belongs.

You will be given:
1. A text to classify
2. A list of available cluster concepts with their titles and keywords

Analyze the text and determine which cluster best matches the content.

Your response must be ONLY a JSON object with the following structure:
{
    "cluster_id": <integer cluster number>,
    "confidence": <float between 0.0 and 1.0>
}

Choose the cluster that best fits the content. Be decisive."""`,

    // scripts/pretrain.py
    pretrain: `import os
import time
import json
from datetime import datetime

import torch
from torch import optim
import torch.nn.functional as F

from mark.augment import feature_dropout, edge_dropout
from mark.backbones.magi import MAGIEncoder
from mark.backbones.dmon import DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all
from mark.utils import set_seed, load_config, setup_logger, get_device


def main():
    # Load configurations
    data_cfg = load_config("configs/data.yaml")
    engine_cfg = load_config("configs/engine.yaml")
    train_cfg = load_config("configs/train.yaml")
    
    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "cpu"))
    logger = setup_logger()

    # Load dataset
    dataset = data_cfg.get("dataset", "cora")
    x, edge_index, texts, y, K = load_tag(dataset, data_cfg.get("data_dir", "./data"))
    
    # Initialize model
    backbone_name = engine_cfg.get("backbone", "magi").lower()
    if backbone_name == "magi":
        model = MAGIEncoder(
            in_dim=x.size(1),
            hidden_dim=engine_cfg.get("hidden_dim", 256),
            proj_dim=engine_cfg.get("proj_dim", 128),
            num_clusters=K,
            tau_align=engine_cfg.get("tau_align", 0.5),
            lambda_clu=engine_cfg.get("lambda_clu", 1.0),
        )
    else:
        model = DMoNEncoder(
            in_dim=x.size(1),
            hidden_dim=engine_cfg.get("hidden_dim", 256),
            num_clusters=K,
        )
    
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.get("lr", 1e-3))
    
    # Training loop
    for epoch in range(1, train_cfg.get("pretrain_epochs", 100) + 1):
        model.train()
        optimizer.zero_grad()
        
        # Create augmented views
        x1 = feature_dropout(x, p=0.2)
        x2 = feature_dropout(x, p=0.2)
        e1 = edge_dropout(edge_index, drop=0.2)
        e2 = edge_dropout(edge_index, drop=0.2)
        
        # Forward pass
        if backbone_name == "magi":
            z1, z2 = model(x1, e1), model(x2, e2)
            loss, comps = model.loss_pretrain(z1, z2)
        else:
            h1, S1 = model(x1, e1)
            h2, S2 = model(x2, e2)
            loss1, _ = model.loss_pretrain(S1, e1)
            loss2, _ = model.loss_pretrain(S2, e2)
            loss = 0.5 * (loss1 + loss2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")

    # Save checkpoint
    torch.save({"model_state": model.state_dict()}, f"checkpoints/{dataset}-{backbone_name}.pt")
    logger.info("Pretraining complete!")


if __name__ == "__main__":
    main()`,

    // scripts/run_mark.py
    run_mark: `import os
import time
import json
from typing import Dict, List, Set

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from mark.augment import feature_dropout, edge_dropout
from mark.backbones import MAGIEncoder, DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all
from mark.agents import LLMClient, induce_concepts, synthesize_for_S, classify_consistency
from mark.ranking import calibration_loss
from mark.utils import set_seed, load_config, setup_logger, get_device


def select_uncertain_nodes(model, x, edge_index, backbone_name):
    """Find uncertain nodes via view disagreement."""
    model.eval()
    with torch.no_grad():
        x1 = feature_dropout(x, 0.1)
        x2 = feature_dropout(x, 0.1)
        e1 = edge_dropout(edge_index, 0.1)
        e2 = edge_dropout(edge_index, 0.1)
        
        if backbone_name == "magi":
            c1 = model.predict_assign(model(x1, e1))
            c2 = model.predict_assign(model(x2, e2))
        else:
            _, S1 = model(x1, e1)
            _, S2 = model(x2, e2)
            c1 = S1.argmax(dim=1)
            c2 = S2.argmax(dim=1)
        
        # Nodes where assignments disagree
        S_set = (c1 != c2).nonzero(as_tuple=False).view(-1)
    
    return S_set


def main():
    # Load configs and initialize
    data_cfg = load_config("configs/data.yaml")
    engine_cfg = load_config("configs/engine.yaml")
    agents_cfg = load_config("configs/agents.yaml")
    train_cfg = load_config("configs/train.yaml")
    
    logger = setup_logger()
    device = get_device(train_cfg.get("device", "cpu"))
    
    # Load data and model
    x, edge_index, texts, y, K = load_tag(data_cfg["dataset"], data_cfg["data_dir"])
    # ... load pretrained model ...
    
    # Initialize LLM client
    llm = LLMClient(
        base_url=agents_cfg["llm"]["base_url"],
        model=agents_cfg["llm"]["model"],
    )
    
    # MARK training loop
    for step in range(1, train_cfg.get("T", 3) + 1):
        # Step 1: Find uncertain nodes
        S = select_uncertain_nodes(model, x, edge_index, backbone_name)
        logger.info(f"[Step {step}] Found {len(S)} uncertain nodes")
        
        # Step 2: Concept Agent
        concepts = induce_concepts(H, assign, top_n=50, client=llm, texts=texts)
        
        # Step 3: Generation Agent
        summaries = synthesize_for_S(S.tolist(), neighbors, texts, concepts, k=10, client=llm)
        
        # Step 4: Inference Agent (consistency filtering)
        R, stats, pred_labels = classify_consistency(
            S.tolist(), {i: texts[i] for i in range(len(texts))},
            summaries, concepts, K, llm
        )
        
        logger.info(f"[Step {step}] Reliable nodes: {len(R)}")
        
        # Step 5: Fine-tune with calibration loss
        for epoch in range(train_cfg.get("ft_epochs_per_step", 10)):
            # ... fine-tuning with L_eng + L_cal ...
            pass
        
        # Evaluate
        metrics = compute_all(y.numpy(), predictions)
        logger.info(f"[Step {step}] ACC={metrics['ACC']:.4f}")


if __name__ == "__main__":
    main()`,

    // scripts/eval.py
    eval: `import os
import json
import torch

from mark.backbones import MAGIEncoder, DMoNEncoder
from mark.datasets import load_tag
from mark.metrics import compute_all, save_metrics
from mark.utils import load_config, setup_logger, get_device


def main():
    """Evaluate trained MARK model."""
    data_cfg = load_config("configs/data.yaml")
    engine_cfg = load_config("configs/engine.yaml")
    train_cfg = load_config("configs/train.yaml")
    
    logger = setup_logger()
    device = get_device(train_cfg.get("device", "cpu"))
    
    # Load dataset
    dataset = data_cfg.get("dataset", "cora")
    x, edge_index, texts, y, K = load_tag(dataset, data_cfg.get("data_dir", "./data"))
    
    # Load model
    backbone_name = engine_cfg.get("backbone", "magi").lower()
    if backbone_name == "magi":
        model = MAGIEncoder(
            in_dim=x.size(1),
            hidden_dim=engine_cfg.get("hidden_dim", 256),
            proj_dim=engine_cfg.get("proj_dim", 128),
            num_clusters=K,
        )
    else:
        model = DMoNEncoder(
            in_dim=x.size(1),
            hidden_dim=engine_cfg.get("hidden_dim", 256),
            num_clusters=K,
        )
    
    # Load checkpoint
    ckpt_path = f"experiments/checkpoints/{dataset}-{backbone_name}-mark.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    
    # Get predictions
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        if backbone_name == "magi":
            z = model(x, edge_index)
            pred = model.predict_assign(z).cpu().numpy()
        else:
            h, S = model(x, edge_index)
            pred = S.argmax(dim=1).cpu().numpy()
    
    # Compute metrics
    metrics = compute_all(y.numpy(), pred)
    
    logger.info(f"=== Evaluation Results for {dataset} ===")
    logger.info(f"ACC: {metrics['ACC']:.4f}")
    logger.info(f"NMI: {metrics['NMI']:.4f}")
    logger.info(f"ARI: {metrics['ARI']:.4f}")
    logger.info(f"F1:  {metrics['F1']:.4f}")
    
    # Save metrics
    save_metrics(metrics, "experiments/metrics.json", "experiments/metrics.csv")
    logger.info("Metrics saved to experiments/metrics.json")


if __name__ == "__main__":
    main()`,

    // configs/data.yaml
    data_yaml: `# Data Configuration

dataset: cora          # Options: cora, citeseer, pubmed, wikics
data_dir: ./data       # Directory to store datasets

# PLM for text encoding
plm_model: sentence-transformers/all-MiniLM-L6-v2
batch_size_plm: 64`,

    // configs/engine.yaml
    engine_yaml: `# Engine Configuration

backbone: magi        # Options: magi, dmon
hidden_dim: 256       # Hidden layer dimension
proj_dim: 128         # Projection dimension (MAGI only)
num_clusters: 0       # 0 = infer from dataset labels

# Loss weights
tau_align: 0.5        # Temperature for NT-Xent (MAGI)
lambda_clu: 0.5       # Weight for clustering loss (MAGI)
t_rank: 0.1           # Temperature for calibration loss

# Architecture
num_layers: 2         # Number of GCN layers
dropout: 0.1          # Dropout rate

# Augmentation
feat_drop: 0.2        # Feature dropout probability
edge_drop: 0.2        # Edge dropout probability

# DMoN specific
collapse_reg: 0.1     # Collapse regularization weight`,

    // configs/agents.yaml
    agents_yaml: `# LLM Agents Configuration

llm:
  base_url: http://localhost:8000/v1
  model: openai/gpt-oss-120b
  temperature: 0.2
  max_tokens: 512

# Concept Agent
top_n_concept: 100    # Top-N samples per cluster for concept induction

# Generation Agent  
k_neighbors: 10       # Number of neighbors for synthesis

# API settings
batch_nodes: 16       # Batch size for LLM calls
concurrency: 2        # Concurrent API requests
retries: 3            # Max retries on failure
retry_backoff: 1.5    # Backoff multiplier`,

    // configs/train.yaml
    train_yaml: `# Training Configuration

device: cuda          # Options: cuda, cpu
amp: true             # Automatic Mixed Precision
seed: 42              # Random seed

# Optimizer
lr: 0.001             # Learning rate
weight_decay: 0.0005  # L2 regularization

# Pretraining
pretrain_epochs: 100  # Number of pretraining epochs

# MARK loop
T: 3                  # Number of MARK steps
T_prime: 1            # Agent collaboration period
ft_epochs_per_step: 10  # Fine-tuning epochs per step

# Paths
log_dir: ./experiments
checkpoint_path: ./experiments/checkpoints`,
};

// ============================================
// Utility Functions
// ============================================

function getCanvasContext(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    // Handle high DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    
    return { ctx, width: rect.width, height: rect.height };
}

function lerp(a, b, t) {
    return a + (b - a) * t;
}

function distance(x1, y1, x2, y2) {
    return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}

// ============================================
// Color Palette
// ============================================

const colors = {
    bg: '#0a0e14',
    bgSecondary: '#0f1419',
    accent: '#f59e0b',
    accentSecondary: '#fbbf24',
    cyan: '#06b6d4',
    emerald: '#10b981',
    rose: '#f43f5e',
    violet: '#8b5cf6',
    textPrimary: '#f8fafc',
    textSecondary: '#94a3b8',
    textTertiary: '#64748b',
};

const clusterColors = [
    '#f59e0b', // amber
    '#06b6d4', // cyan
    '#10b981', // emerald
    '#f43f5e', // rose
    '#8b5cf6', // violet
    '#ec4899', // pink
    '#14b8a6', // teal
];

// ============================================
// Hero Graph Animation
// ============================================

class HeroGraph {
    constructor(canvasId) {
        const result = getCanvasContext(canvasId);
        if (!result) return;
        
        this.ctx = result.ctx;
        this.width = result.width;
        this.height = result.height;
        this.nodes = [];
        this.edges = [];
        this.time = 0;
        
        this.init();
        this.animate();
    }
    
    init() {
        // Create nodes in clusters
        const numClusters = 4;
        const nodesPerCluster = 8;
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const clusterRadius = Math.min(this.width, this.height) * 0.3;
        
        for (let c = 0; c < numClusters; c++) {
            const angle = (c / numClusters) * Math.PI * 2 - Math.PI / 2;
            const cx = centerX + Math.cos(angle) * clusterRadius;
            const cy = centerY + Math.sin(angle) * clusterRadius;
            
            for (let i = 0; i < nodesPerCluster; i++) {
                const nodeAngle = Math.random() * Math.PI * 2;
                const nodeRadius = Math.random() * 60 + 20;
                
                this.nodes.push({
                    x: cx + Math.cos(nodeAngle) * nodeRadius,
                    y: cy + Math.sin(nodeAngle) * nodeRadius,
                    baseX: cx + Math.cos(nodeAngle) * nodeRadius,
                    baseY: cy + Math.sin(nodeAngle) * nodeRadius,
                    radius: Math.random() * 4 + 4,
                    cluster: c,
                    phase: Math.random() * Math.PI * 2,
                });
            }
        }
        
        // Create edges within clusters and some between
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dist = distance(
                    this.nodes[i].baseX, this.nodes[i].baseY,
                    this.nodes[j].baseX, this.nodes[j].baseY
                );
                
                // Connect within same cluster
                if (this.nodes[i].cluster === this.nodes[j].cluster && dist < 100) {
                    this.edges.push({ from: i, to: j, strength: 1 });
                }
                // Occasional cross-cluster connections
                else if (dist < 150 && Math.random() < 0.1) {
                    this.edges.push({ from: i, to: j, strength: 0.3 });
                }
            }
        }
    }
    
    animate() {
        this.time += 0.005;
        
        // Update node positions with gentle floating
        for (const node of this.nodes) {
            node.x = node.baseX + Math.sin(this.time * 2 + node.phase) * 3;
            node.y = node.baseY + Math.cos(this.time * 2 + node.phase * 1.3) * 3;
        }
        
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
    
    draw() {
        const { ctx, width, height } = this;
        
        // Clear
        ctx.clearRect(0, 0, width, height);
        
        // Draw edges
        for (const edge of this.edges) {
            const from = this.nodes[edge.from];
            const to = this.nodes[edge.to];
            
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = `rgba(148, 163, 184, ${0.15 * edge.strength})`;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        
        // Draw nodes
        for (const node of this.nodes) {
            // Glow effect
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius * 3
            );
            gradient.addColorStop(0, `${clusterColors[node.cluster]}40`);
            gradient.addColorStop(1, 'transparent');
            
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * 3, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
            
            // Node
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = clusterColors[node.cluster];
            ctx.fill();
        }
    }
}

// ============================================
// Graph Basics Visualization
// ============================================

class GraphBasicsViz {
    constructor(canvasId) {
        const result = getCanvasContext(canvasId);
        if (!result) return;
        
        this.ctx = result.ctx;
        this.width = result.width;
        this.height = result.height;
        this.nodes = [];
        this.edges = [];
        this.currentGraph = 'simple';
        this.time = 0;
        
        this.createGraph('simple');
        this.animate();
        this.setupControls();
    }
    
    setupControls() {
        const buttons = document.querySelectorAll('.viz-btn[data-graph]');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                buttons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.createGraph(btn.dataset.graph);
            });
        });
    }
    
    createGraph(type) {
        this.nodes = [];
        this.edges = [];
        this.currentGraph = type;
        
        const cx = this.width / 2;
        const cy = this.height / 2;
        
        if (type === 'simple') {
            // Simple 5-node graph
            const positions = [
                { x: cx, y: cy - 80 },
                { x: cx - 100, y: cy },
                { x: cx + 100, y: cy },
                { x: cx - 60, y: cy + 80 },
                { x: cx + 60, y: cy + 80 },
            ];
            
            positions.forEach((pos, i) => {
                this.nodes.push({
                    x: pos.x,
                    y: pos.y,
                    baseX: pos.x,
                    baseY: pos.y,
                    radius: 15,
                    label: i.toString(),
                    color: colors.accent,
                    phase: Math.random() * Math.PI * 2,
                });
            });
            
            this.edges = [
                { from: 0, to: 1 },
                { from: 0, to: 2 },
                { from: 1, to: 3 },
                { from: 2, to: 4 },
                { from: 3, to: 4 },
            ];
        } else if (type === 'citation') {
            // Citation network pattern
            const layers = [
                [{ x: cx, y: cy - 100 }],
                [{ x: cx - 80, y: cy - 30 }, { x: cx + 80, y: cy - 30 }],
                [{ x: cx - 120, y: cy + 50 }, { x: cx, y: cy + 50 }, { x: cx + 120, y: cy + 50 }],
            ];
            
            let idx = 0;
            const nodeColors = [colors.rose, colors.cyan, colors.cyan, colors.emerald, colors.emerald, colors.emerald];
            
            layers.forEach(layer => {
                layer.forEach(pos => {
                    this.nodes.push({
                        x: pos.x,
                        y: pos.y,
                        baseX: pos.x,
                        baseY: pos.y,
                        radius: 18,
                        label: `📄`,
                        color: nodeColors[idx],
                        phase: Math.random() * Math.PI * 2,
                    });
                    idx++;
                });
            });
            
            // Citation edges (top cites bottom)
            this.edges = [
                { from: 1, to: 0 },
                { from: 2, to: 0 },
                { from: 3, to: 1 },
                { from: 4, to: 1 },
                { from: 4, to: 2 },
                { from: 5, to: 2 },
            ];
        } else if (type === 'social') {
            // Social network with clusters
            const clusterCenters = [
                { x: cx - 80, y: cy - 40 },
                { x: cx + 80, y: cy + 40 },
            ];
            
            let idx = 0;
            clusterCenters.forEach((center, ci) => {
                // 4 nodes per cluster
                for (let i = 0; i < 4; i++) {
                    const angle = (i / 4) * Math.PI * 2;
                    const r = 50;
                    this.nodes.push({
                        x: center.x + Math.cos(angle) * r,
                        y: center.y + Math.sin(angle) * r,
                        baseX: center.x + Math.cos(angle) * r,
                        baseY: center.y + Math.sin(angle) * r,
                        radius: 14,
                        label: '👤',
                        color: ci === 0 ? colors.violet : colors.cyan,
                        phase: Math.random() * Math.PI * 2,
                    });
                    idx++;
                }
            });
            
            // Intra-cluster edges
            for (let i = 0; i < 4; i++) {
                for (let j = i + 1; j < 4; j++) {
                    if (Math.random() < 0.7) {
                        this.edges.push({ from: i, to: j });
                    }
                    if (Math.random() < 0.7) {
                        this.edges.push({ from: i + 4, to: j + 4 });
                    }
                }
            }
            // Bridge
            this.edges.push({ from: 2, to: 4 });
        }
    }
    
    animate() {
        this.time += 0.01;
        
        for (const node of this.nodes) {
            node.x = node.baseX + Math.sin(this.time + node.phase) * 2;
            node.y = node.baseY + Math.cos(this.time * 1.2 + node.phase) * 2;
        }
        
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
    
    draw() {
        const { ctx, width, height } = this;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw edges
        for (const edge of this.edges) {
            const from = this.nodes[edge.from];
            const to = this.nodes[edge.to];
            
            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        // Draw nodes
        for (const node of this.nodes) {
            // Glow
            const gradient = ctx.createRadialGradient(
                node.x, node.y, 0,
                node.x, node.y, node.radius * 2
            );
            gradient.addColorStop(0, `${node.color}40`);
            gradient.addColorStop(1, 'transparent');
            
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius * 2, 0, Math.PI * 2);
            ctx.fillStyle = gradient;
            ctx.fill();
            
            // Node circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            ctx.fillStyle = '#1a222d';
            ctx.strokeStyle = node.color;
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            // Label
            ctx.fillStyle = colors.textPrimary;
            ctx.font = node.label.length === 1 ? 'bold 12px JetBrains Mono' : '14px serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.label, node.x, node.y);
        }
    }
}

// ============================================
// Message Passing Visualization
// ============================================

class MessagePassingViz {
    constructor() {
        this.initGatherViz();
        this.initAggregateViz();
        this.initUpdateViz();
    }
    
    initGatherViz() {
        const result = getCanvasContext('gatherCanvas');
        if (!result) return;
        
        const { ctx, width, height } = result;
        let time = 0;
        
        const centerNode = { x: width / 2, y: height / 2 };
        const neighborNodes = [];
        
        for (let i = 0; i < 4; i++) {
            const angle = (i / 4) * Math.PI * 2 - Math.PI / 2;
            neighborNodes.push({
                x: centerNode.x + Math.cos(angle) * 40,
                y: centerNode.y + Math.sin(angle) * 40,
            });
        }
        
        const animate = () => {
            time += 0.02;
            ctx.clearRect(0, 0, width, height);
            
            // Draw edges with animated messages
            for (let i = 0; i < neighborNodes.length; i++) {
                const n = neighborNodes[i];
                ctx.beginPath();
                ctx.moveTo(n.x, n.y);
                ctx.lineTo(centerNode.x, centerNode.y);
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
                
                // Animated message dot
                const t = (Math.sin(time + i * 0.5) + 1) / 2;
                const mx = lerp(n.x, centerNode.x, t);
                const my = lerp(n.y, centerNode.y, t);
                
                ctx.beginPath();
                ctx.arc(mx, my, 3, 0, Math.PI * 2);
                ctx.fillStyle = colors.cyan;
                ctx.fill();
            }
            
            // Draw neighbor nodes
            for (const n of neighborNodes) {
                ctx.beginPath();
                ctx.arc(n.x, n.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = colors.cyan;
                ctx.fill();
            }
            
            // Draw center node
            ctx.beginPath();
            ctx.arc(centerNode.x, centerNode.y, 12, 0, Math.PI * 2);
            ctx.fillStyle = colors.accent;
            ctx.fill();
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    initAggregateViz() {
        const result = getCanvasContext('aggregateCanvas');
        if (!result) return;
        
        const { ctx, width, height } = result;
        let time = 0;
        
        const animate = () => {
            time += 0.03;
            ctx.clearRect(0, 0, width, height);
            
            // Draw converging arrows
            const cx = width / 2;
            const cy = height / 2;
            
            for (let i = 0; i < 4; i++) {
                const angle = (i / 4) * Math.PI * 2;
                const r = 35 + Math.sin(time * 2) * 5;
                
                ctx.beginPath();
                ctx.moveTo(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r);
                ctx.lineTo(cx, cy);
                ctx.strokeStyle = colors.emerald;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            
            // Aggregation symbol (sum)
            ctx.beginPath();
            ctx.arc(cx, cy, 20 + Math.sin(time * 3) * 3, 0, Math.PI * 2);
            ctx.fillStyle = `${colors.emerald}30`;
            ctx.strokeStyle = colors.emerald;
            ctx.lineWidth = 2;
            ctx.fill();
            ctx.stroke();
            
            ctx.fillStyle = colors.textPrimary;
            ctx.font = 'bold 16px JetBrains Mono';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Σ', cx, cy);
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    initUpdateViz() {
        const result = getCanvasContext('updateCanvas');
        if (!result) return;
        
        const { ctx, width, height } = result;
        let time = 0;
        
        const animate = () => {
            time += 0.02;
            ctx.clearRect(0, 0, width, height);
            
            const cx = width / 2;
            const cy = height / 2;
            
            // Neural network layers
            const layers = [
                [{ y: cy - 20 }, { y: cy + 20 }],
                [{ y: cy - 15 }, { y: cy }, { y: cy + 15 }],
                [{ y: cy }],
            ];
            
            const layerX = [cx - 30, cx, cx + 30];
            
            // Draw connections
            ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
            ctx.lineWidth = 1;
            
            for (let l = 0; l < layers.length - 1; l++) {
                for (const from of layers[l]) {
                    for (const to of layers[l + 1]) {
                        ctx.beginPath();
                        ctx.moveTo(layerX[l], from.y);
                        ctx.lineTo(layerX[l + 1], to.y);
                        ctx.stroke();
                    }
                }
            }
            
            // Draw neurons
            for (let l = 0; l < layers.length; l++) {
                for (const node of layers[l]) {
                    const pulse = Math.sin(time * 3 + l * 0.5) * 2;
                    ctx.beginPath();
                    ctx.arc(layerX[l], node.y, 6 + pulse, 0, Math.PI * 2);
                    ctx.fillStyle = l === layers.length - 1 ? colors.accent : colors.violet;
                    ctx.fill();
                }
            }
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
}

// ============================================
// Contrastive Learning Visualization
// ============================================

class ContrastiveViz {
    constructor() {
        this.initView('view1Canvas', 0);
        this.initView('view2Canvas', Math.PI);
    }
    
    initView(canvasId, phaseOffset) {
        const result = getCanvasContext(canvasId);
        if (!result) return;
        
        const { ctx, width, height } = result;
        let time = 0;
        
        // Create a small graph
        const nodes = [];
        const edges = [];
        const cx = width / 2;
        const cy = height / 2;
        
        for (let i = 0; i < 6; i++) {
            const angle = (i / 6) * Math.PI * 2;
            const r = 50;
            nodes.push({
                x: cx + Math.cos(angle) * r,
                y: cy + Math.sin(angle) * r,
                baseX: cx + Math.cos(angle) * r,
                baseY: cy + Math.sin(angle) * r,
                visible: true,
                phase: i * 0.5,
            });
        }
        
        // All connected
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                edges.push({ from: i, to: j, visible: true });
            }
        }
        
        const animate = () => {
            time += 0.02;
            ctx.clearRect(0, 0, width, height);
            
            // Simulate augmentation: some edges/features dropped
            const dropPhase = (time + phaseOffset) % (Math.PI * 2);
            
            // Draw edges (some dropped)
            for (let i = 0; i < edges.length; i++) {
                const edge = edges[i];
                const visible = Math.sin(dropPhase + i * 0.3) > -0.5;
                
                if (visible) {
                    const from = nodes[edge.from];
                    const to = nodes[edge.to];
                    
                    ctx.beginPath();
                    ctx.moveTo(from.x, from.y);
                    ctx.lineTo(to.x, to.y);
                    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
            
            // Draw nodes (with varying opacity for feature dropout)
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                const featureStrength = 0.5 + 0.5 * Math.sin(dropPhase + node.phase);
                
                node.x = node.baseX + Math.sin(time + node.phase) * 2;
                node.y = node.baseY + Math.cos(time * 1.2 + node.phase) * 2;
                
                ctx.beginPath();
                ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(245, 158, 11, ${0.3 + featureStrength * 0.7})`;
                ctx.fill();
            }
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
}

// ============================================
// Clustering Comparison Visualization
// ============================================

class ClusteringViz {
    constructor() {
        this.initTraditional();
        this.initDeep();
    }
    
    initTraditional() {
        const result = getCanvasContext('traditionalClusterViz');
        if (!result) return;
        
        const { ctx, width, height } = result;
        let time = 0;
        
        // Two clusters based only on features (y-axis position)
        const nodes = [];
        for (let i = 0; i < 20; i++) {
            const cluster = i < 10 ? 0 : 1;
            nodes.push({
                x: Math.random() * (width - 40) + 20,
                y: cluster === 0 ? 30 + Math.random() * 60 : 110 + Math.random() * 60,
                cluster,
            });
        }
        
        const animate = () => {
            time += 0.02;
            ctx.clearRect(0, 0, width, height);
            
            // Draw cluster boundaries
            ctx.fillStyle = 'rgba(6, 182, 212, 0.1)';
            ctx.fillRect(0, 0, width, height / 2);
            ctx.fillStyle = 'rgba(245, 158, 11, 0.1)';
            ctx.fillRect(0, height / 2, width, height / 2);
            
            // Dashed dividing line
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.strokeStyle = colors.textTertiary;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.setLineDash([]);
            
            // Draw nodes
            for (const node of nodes) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, 5, 0, Math.PI * 2);
                ctx.fillStyle = node.cluster === 0 ? colors.cyan : colors.accent;
                ctx.fill();
            }
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    initDeep() {
        const result = getCanvasContext('deepClusterViz');
        if (!result) return;
        
        const { ctx, width, height } = result;
        let time = 0;
        
        // Clusters based on graph structure
        const clusters = [
            { cx: width * 0.3, cy: height * 0.4, color: colors.cyan },
            { cx: width * 0.7, cy: height * 0.6, color: colors.accent },
        ];
        
        const nodes = [];
        const edges = [];
        
        clusters.forEach((cluster, ci) => {
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * Math.PI * 2;
                const r = 25 + Math.random() * 20;
                nodes.push({
                    x: cluster.cx + Math.cos(angle) * r,
                    y: cluster.cy + Math.sin(angle) * r,
                    baseX: cluster.cx + Math.cos(angle) * r,
                    baseY: cluster.cy + Math.sin(angle) * r,
                    cluster: ci,
                    phase: Math.random() * Math.PI * 2,
                });
            }
        });
        
        // Intra-cluster edges
        for (let c = 0; c < 2; c++) {
            const start = c * 8;
            for (let i = 0; i < 8; i++) {
                for (let j = i + 1; j < 8; j++) {
                    if (Math.random() < 0.4) {
                        edges.push({ from: start + i, to: start + j });
                    }
                }
            }
        }
        // One bridge edge
        edges.push({ from: 4, to: 12 });
        
        const animate = () => {
            time += 0.015;
            ctx.clearRect(0, 0, width, height);
            
            // Update positions
            for (const node of nodes) {
                node.x = node.baseX + Math.sin(time + node.phase) * 2;
                node.y = node.baseY + Math.cos(time * 1.2 + node.phase) * 2;
            }
            
            // Draw edges
            for (const edge of edges) {
                const from = nodes[edge.from];
                const to = nodes[edge.to];
                
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            
            // Draw nodes
            for (const node of nodes) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, 5, 0, Math.PI * 2);
                ctx.fillStyle = clusters[node.cluster].color;
                ctx.fill();
            }
            
            requestAnimationFrame(animate);
        };
        
        animate();
    }
}

// ============================================
// Results Chart
// ============================================

function initResultsChart() {
    const ctx = document.getElementById('resultsChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Cora', 'CiteSeer', 'PubMed', 'WikiCS'],
            datasets: [
                {
                    label: 'MAGI (Baseline)',
                    data: [70.2, 65.8, 68.4, 52.1],
                    backgroundColor: 'rgba(148, 163, 184, 0.5)',
                    borderColor: 'rgba(148, 163, 184, 1)',
                    borderWidth: 1,
                },
                {
                    label: 'MARK (Ours)',
                    data: [78.5, 71.2, 73.9, 58.7],
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderColor: 'rgba(245, 158, 11, 1)',
                    borderWidth: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8',
                        font: {
                            family: 'Plus Jakarta Sans',
                        },
                    },
                },
                title: {
                    display: true,
                    text: 'Clustering Accuracy (%) Comparison',
                    color: '#f8fafc',
                    font: {
                        size: 16,
                        family: 'Plus Jakarta Sans',
                        weight: '600',
                    },
                },
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)',
                    },
                },
                y: {
                    beginAtZero: false,
                    min: 40,
                    max: 85,
                    ticks: {
                        color: '#94a3b8',
                        callback: (value) => value + '%',
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)',
                    },
                },
            },
        },
    });
}

// ============================================
// Code Syntax Highlighting
// ============================================

function initSyntaxHighlighting() {
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
}

// ============================================
// Smooth Scroll & Navbar
// ============================================

function initNavigation() {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                });
            }
        });
    });
    
    // Navbar background on scroll
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.style.background = 'rgba(10, 14, 20, 0.95)';
            } else {
                navbar.style.background = 'rgba(10, 14, 20, 0.8)';
            }
        });
    }
}

// ============================================
// File Tree Toggle & Code Viewer
// ============================================

function initFileTree() {
    // Folder toggle
    document.querySelectorAll('.tree-item.folder').forEach(folder => {
        folder.addEventListener('click', function(e) {
            if (e.target === this || e.target.classList.contains('folder-icon')) {
                this.classList.toggle('open');
                e.stopPropagation();
            }
        });
    });
    
    // File click to open modal
    document.querySelectorAll('.tree-item.file[data-file]').forEach(file => {
        file.addEventListener('click', function(e) {
            e.stopPropagation();
            const fileKey = this.dataset.file;
            const fileName = this.querySelector('.file-name')?.textContent || fileKey;
            openCodeModal(fileKey, fileName);
        });
    });
}

function openCodeModal(fileKey, fileName) {
    const modal = document.getElementById('codeModal');
    const title = document.getElementById('codeModalTitle');
    const content = document.getElementById('codeModalContent');
    
    if (!modal || !title || !content) return;
    
    const code = FILE_CONTENTS[fileKey];
    if (!code) {
        console.warn(`No content found for file: ${fileKey}`);
        return;
    }
    
    title.textContent = fileName;
    content.textContent = code;
    
    // Determine language for syntax highlighting
    const lang = fileName.endsWith('.yaml') ? 'yaml' : 'python';
    content.className = `language-${lang}`;
    
    // Re-apply syntax highlighting
    if (typeof hljs !== 'undefined') {
        hljs.highlightElement(content);
    }
    
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeCodeModal() {
    const modal = document.getElementById('codeModal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

// ============================================
// Citation Modal
// ============================================

function initCitationModal() {
    const citationLink = document.getElementById('citation-link');
    const citationModal = document.getElementById('citationModal');
    const closeBtn = citationModal?.querySelector('.citation-modal-close');
    const copyBtn = citationModal?.querySelector('.copy-citation');
    
    if (citationLink && citationModal) {
        citationLink.addEventListener('click', (e) => {
            e.preventDefault();
            citationModal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        });
        
        closeBtn?.addEventListener('click', () => {
            citationModal.style.display = 'none';
            document.body.style.overflow = '';
        });
        
        citationModal.addEventListener('click', (e) => {
            if (e.target === citationModal) {
                citationModal.style.display = 'none';
                document.body.style.overflow = '';
            }
        });
        
        copyBtn?.addEventListener('click', () => {
            const citationText = document.getElementById('citationText')?.textContent;
            if (citationText) {
                navigator.clipboard.writeText(citationText).then(() => {
                    copyBtn.textContent = '✓ Copied!';
                    setTimeout(() => {
                        copyBtn.textContent = 'Copy to Clipboard';
                    }, 2000);
                });
            }
        });
    }
}

// ============================================
// Update GitHub Links from Config
// ============================================

function updateGitHubLinks() {
    // Update all GitHub links with the configured URL
    document.querySelectorAll('.github-link, #github-link, .nav-github').forEach(link => {
        if (link.href && (link.href.includes('github.com/yourusername') || link.href === 'https://github.com')) {
            link.href = CONFIG.GITHUB_REPO;
        }
    });
    
    // Update paper link
    const paperLink = document.getElementById('paper-link');
    if (paperLink) {
        paperLink.href = CONFIG.PAPER_URL;
    }
}

// ============================================
// Initialize Everything
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Update links from config
    updateGitHubLinks();
    
    // Initialize visualizations
    new HeroGraph('heroGraph');
    new GraphBasicsViz('graphBasicsViz');
    new MessagePassingViz();
    new ContrastiveViz();
    new ClusteringViz();
    
    // Initialize chart
    initResultsChart();
    
    // Initialize code highlighting
    initSyntaxHighlighting();
    
    // Initialize navigation
    initNavigation();
    
    // Initialize file tree with code viewer
    initFileTree();
    
    // Initialize citation modal
    initCitationModal();
    
    // Code modal close handlers
    const codeModalClose = document.getElementById('codeModalClose');
    const codeModal = document.getElementById('codeModal');
    
    if (codeModalClose) {
        codeModalClose.addEventListener('click', closeCodeModal);
    }
    
    if (codeModal) {
        codeModal.addEventListener('click', (e) => {
            if (e.target === codeModal) {
                closeCodeModal();
            }
        });
    }
    
    // Close modals on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeCodeModal();
            const citationModal = document.getElementById('citationModal');
            if (citationModal) {
                citationModal.style.display = 'none';
                document.body.style.overflow = '';
            }
        }
    });
    
    console.log('MARK Blog initialized');
});

// Handle window resize
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        // Reinitialize visualizations on resize
        // This is a simple approach; for production, you'd want more sophisticated handling
    }, 250);
});

