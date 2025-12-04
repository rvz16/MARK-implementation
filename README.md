# MARK: Multi-Agent Graph Clustering with GNN Backbones

The foundation and reproducibility are based on the attached paper 2025.findings-acl.314.pdf: implemented end-to-end cycle pretrain → agents → filter R → update X → ranking fine-tuning → metrics over GNN backbones (MAGI, DMoN) with local LLM.

## Interactive Blog

We provide an interactive blog that explains Graph ML concepts and the MARK framework step-by-step:

**[View the Blog](https://rvz16.github.io/MARK-implementation/)**

### Running the Blog Locally
```bash

# Option 1: Python HTTP server
cd docs && python -m http.server 8080
# Then open http://localhost:8080

# Option 2: Node.js (if installed)
npx serve docs

# Option 3: VS Code Live Server extension
# Right-click docs/index.html → "Open with Live Server"
```

### Deploying to GitHub Pages
1. Go to your repository Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select `main` branch and `/docs` folder
4. Save and wait for deployment

The blog covers:
- Graph ML fundamentals (nodes, edges, features)
- PyTorch Geometric (PyG) basics with code examples
- Graph Neural Networks and message passing
- Deep graph clustering (MAGI, DMoN backbones)
- The MARK multi-agent framework
- Complete implementation walkthrough

## Installation
- Install PyTorch with appropriate CUDA/CPU build: see https://pytorch.org/get-started/locally/
- `pip install -r requirements.txt`
- Export data directory: `set TORCH_GEOMETRIC_HOME=./data` (Windows PowerShell) or `export TORCH_GEOMETRIC_HOME=./data`
- If necessary, specify LLM endpoint/model in `configs/agents.yaml` (default: `http://10.100.10.70:9999/v1`, `openai/gpt-oss-120b`).

## Structure
- `mark/`: package with data loading, augmentations, metrics, utilities, backbones (MAGI, DMoN), agents and ranking loss.
- `configs/`: `data.yaml`, `engine.yaml`, `agents.yaml`, `train.yaml`.
- `scripts/`: `check_data.py`, `pretrain.py`, `run_mark.py`, `eval.py`, `ablations.py`, `sensitivity.py`.
- `experiments/`: logs, checkpoints, caches (has `.gitkeep`).

## Quick Start
```bash
python scripts/check_data.py
python scripts/pretrain.py
python scripts/run_mark.py
python scripts/eval.py
```
- All paths and hyperparameters are taken from `configs/*.yaml`.
- Pretrain checkpoint is copied to `experiments/checkpoints/<dataset>-<backbone>-pretrain.pt`.

## Config Parameters
- `configs/data.yaml`: `dataset (cora|citeseer|pubmed|wikics)`, `data_dir`, `plm_model` (SentenceTransformer), `batch_size_plm`.
- `configs/engine.yaml`: `backbone (magi|dmon)`, `hidden_dim`, `proj_dim`, `num_clusters (0=from dataset)`, `tau_align`, `lambda_clu`, `t_rank` (temperature/weight calibration).
- `configs/agents.yaml`: `llm.base_url`, `llm.model`, `temperature`, `max_tokens`, `top_n_concept`, `k_neighbors`, `batch_nodes`, `concurrency`, `retries`, `retry_backoff`.
- `configs/train.yaml`: `device`, `amp`, `seed`, `lr`, `weight_decay`, `T` (number of agent+FT steps), `T_prime` (agent period), `ft_epochs_per_step`, `log_dir`, `pretrain_epochs`, `checkpoint_path`.

## Typical Pipeline
1. **Data Loading** (`check_data.py`): downloading TAG (Cora by default), printing sizes and internet check.
2. **Pretrain** (`pretrain.py`): self-supervised MAGI (NT-Xent + clustering) or DMoN (modularity) with two augmentations; checkpoint, centers, disagreements S are saved.
3. **MARK Cycle** (`run_mark.py`): every `T_prime` epochs
   - Concept induction (LLM, `agents/concept.py`)
   - Synthetic generation for S (LLM)
   - Consistency inference → set R
   - Feature update by averaging with PLM embeddings of summarized nodes
   - Fine-tuning with ranking calibration `Lft = Leng + Lcal`
   - Time logs (`timing_report.json`), tokens (`costs.json`), metrics per steps.
4. **Evaluation** (`eval.py`): ACC/NMI/ARI/F1 in `metrics.json`/`metrics.csv`.
5. **Ablations/Sensitivity**: `ablations.py` (disabling concept/generation/inference) and `sensitivity.py` (grids top-n, k, plots `sensitivity.png`).

## Notes
- All agents require local LLM with OpenAI-compatible API; concurrency, batches and retries are managed in `agents.yaml`.
- PLM caches and agent responses are stored in `experiments/<run>/`.
- AMP is enabled when `device=cuda` and `amp=true`.

## Readiness Check
- After `pretrain.py` and one step of `run_mark.py` on Cora should appear: non-empty set S, non-zero R, decreasing `Lali` (MAGI) and ACC/NMI/ARI/F1 metrics in `metrics_step*.json` and `scripts/eval.py`.
