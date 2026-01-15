# Web of Microbes

Multi-modal foundation model for soil microbiome substrate preference prediction.

## Components

| Component | Description |
|-----------|-------------|
| **Knowledge Graph** | Integrated from KGmicrobe + MetagenomeKG (2.39M nodes) |
| **Language Tower** | Qwen2.5-14B with QLoRA CPT on soil microbiology literature |
| **Substrate Preference (BSPR)** | Bayesian genus-level metabolite preference model |
| **Webapp** | FastAPI + Gradio interface for predictions |

## Models (HuggingFace)

- LLM: `northenlab/soilfm-qwen2.5-14b-literature-cpt`
- KG: `northenlab/soilfm2-graph-tower-joint-v0.1`
- KG Data: `northenlab/webofmicrobes-knowledge-graph` (private)

### Download KG Files

```bash
# Requires HuggingFace access to private repo
huggingface-cli download northenlab/webofmicrobes-knowledge-graph \
  --local-dir knowledge_graph/ --repo-type dataset
```

## Directory Structure

```
├── data/                      # Training data (exometabolomics)
├── knowledge_graph/           # KG construction and data
├── language_tower/            # LLM training scripts
├── substrate_preference/      # BSPR model and training
├── pipeline/                  # Recommender system
├── webapp/                    # Web interface
└── evaluation/                # Validation scripts
```

## Quick Start

1. Start vLLM server:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model northenlab/soilfm-qwen2.5-14b-literature-cpt \
  --host 0.0.0.0 --port 8001 &
```

2. Start webapp:
```bash
PYTHONPATH=./substrate_preference:./pipeline \
  uvicorn webapp.app:app --host 0.0.0.0 --port 8000
```

## Key Metrics

| Model | Metric | Value |
|-------|--------|-------|
| BSPR | AUC | 0.94 |
| BSPR | Genera covered | 20 |
| BSPR | Metabolites | 403 |
| LLM | Validation loss | 1.61 |
| KG | Nodes | 2.39M |

## Citation

Northen Lab, Lawrence Berkeley National Laboratory
