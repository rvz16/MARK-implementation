import os
import json
import urllib.request
from typing import List, Tuple, Optional

import torch
import numpy as np
from torch_geometric.datasets import Planetoid, WikiCS


# URLs for TAG benchmark datasets with raw texts (from Chen et al. 2023)
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
    "citeseer": [
        "Agents", "AI", "DB", "IR", "ML", "HCI"
    ],
    "pubmed": [
        "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2", "Diabetes Mellitus Experimental"
    ],
    "wikics": [
        "Computational linguistics", "Databases", "Operating systems", 
        "Computer architecture", "Computer security", "Internet protocols",
        "Computer file systems", "Distributed computing architecture",
        "Web technology", "Programming language topics"
    ],
}


def _ensure_data_dir(root: str) -> str:
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    os.environ.setdefault("TORCH_GEOMETRIC_HOME", root)
    return root


def _download_raw_texts(dataset_name: str, root: str) -> Optional[List[str]]:
    """Download raw texts for TAG datasets from TAPE repository."""
    dataset_name = dataset_name.lower()
    if dataset_name not in TAG_URLS:
        return None
    
    cache_path = os.path.join(root, f"{dataset_name}_raw_texts.json")
    
    if os.path.exists(cache_path):
        print(f"[DATA] Loading cached raw texts from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle both list and dict formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Convert dict format {node_id: text} to list
            max_id = max(int(k) for k in data.keys())
            texts = [""] * (max_id + 1)
            for k, v in data.items():
                texts[int(k)] = v if isinstance(v, str) else v.get("text", str(v))
            return texts
    
    url = TAG_URLS[dataset_name]
    print(f"[DATA] Downloading raw texts from {url}")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        # Save cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            max_id = max(int(k) for k in data.keys())
            texts = [""] * (max_id + 1)
            for k, v in data.items():
                texts[int(k)] = v if isinstance(v, str) else v.get("text", str(v))
            return texts
    except Exception as e:
        print(f"[DATA] Warning: Could not download raw texts: {e}")
        return None


def _generate_texts_from_features(x: torch.Tensor, y: torch.Tensor, dataset_name: str) -> List[str]:
    """
    Generate synthetic text descriptions from node features.
    This is a fallback when raw texts are not available.
    """
    dataset_name = dataset_name.lower()
    class_names = CLASS_NAMES.get(dataset_name, [f"Class_{i}" for i in range(int(y.max().item()) + 1)])
    
    texts = []
    # Get top feature indices for each node as "keywords"
    x_np = x.numpy()
    
    for i in range(x.size(0)):
        node_features = x_np[i]
        # Get top-10 non-zero feature indices
        top_indices = np.argsort(node_features)[-10:][::-1]
        top_values = node_features[top_indices]
        
        # Filter to non-zero features
        mask = top_values > 0
        top_indices = top_indices[mask][:5]
        
        # Create a synthetic description
        label = int(y[i].item())
        class_name = class_names[label] if label < len(class_names) else f"Class_{label}"
        
        if len(top_indices) > 0:
            feature_str = ", ".join([f"feature_{idx}" for idx in top_indices])
            text = f"A {class_name} paper with key features: {feature_str}. "
            text += f"This document belongs to the {class_name} category."
        else:
            text = f"A {class_name} paper. This document belongs to the {class_name} category."
        
        texts.append(text)
    
    return texts


def load_tag(name: str, root: str) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor, int]:
    """
    Load text-attributed graph datasets.
    Returns:
        X: node features (FloatTensor)
        edge_index: COO format [2, E]
        texts: raw textual descriptions for each node
        y: node labels
        K: number of classes
    """
    root = _ensure_data_dir(root)
    name_l = name.lower()
    
    if name_l in {"cora", "citeseer", "pubmed"}:
        dataset = Planetoid(root=root, name=name_l.capitalize())
    elif name_l == "wikics":
        dataset = WikiCS(root=root)
    else:
        raise ValueError(f"Unsupported dataset {name}")

    data = dataset[0]
    x = data.x.float()
    edge_index = data.edge_index
    y = data.y.long()
    K = int(y.max().item() + 1)
    
    # Try to load raw texts
    texts = _download_raw_texts(name_l, root)
    
    if texts is None or len(texts) != x.size(0):
        print(f"[DATA] Raw texts not available or size mismatch, generating from features")
        texts = _generate_texts_from_features(x, y, name_l)
    else:
        # Validate texts are non-empty strings
        for i, t in enumerate(texts):
            if not t or not isinstance(t, str):
                texts[i] = f"Node {i} document."
    
    num_edges = edge_index.size(1)
    print(f"[DATA] Loaded {name_l} |V|={x.size(0)} |E|={num_edges} K={K} texts_loaded={texts is not None}")
    
    # Print sample text for verification
    if texts:
        print(f"[DATA] Sample text (node 0): {texts[0][:200]}...")
    
    return x, edge_index, texts, y, K


def load_tag_with_plm(
    name: str, 
    root: str, 
    plm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor, int]:
    """
    Load TAG dataset and optionally re-encode features using PLM.
    """
    x, edge_index, texts, y, K = load_tag(name, root)
    
    # Check if we should use PLM encoding for features
    cache_path = os.path.join(root, f"{name.lower()}_plm_features.pt")
    
    if os.path.exists(cache_path):
        print(f"[DATA] Loading PLM features from cache: {cache_path}")
        x_plm = torch.load(cache_path)
        if x_plm.size(0) == x.size(0):
            return x_plm, edge_index, texts, y, K
    
    # Check if texts are meaningful (not just node_i placeholders)
    sample_texts = texts[:10]
    if all(t.startswith("node_") for t in sample_texts):
        print("[DATA] Texts are placeholders, using original features")
        return x, edge_index, texts, y, K
    
    # Encode texts with PLM
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[DATA] Encoding texts with PLM: {plm_model}")
        model = SentenceTransformer(plm_model)
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        x_plm = torch.tensor(embeddings, dtype=torch.float32)
        
        # Cache the PLM features
        torch.save(x_plm, cache_path)
        print(f"[DATA] PLM features saved to {cache_path}")
        
        return x_plm, edge_index, texts, y, K
    except Exception as e:
        print(f"[DATA] PLM encoding failed: {e}, using original features")
        return x, edge_index, texts, y, K
