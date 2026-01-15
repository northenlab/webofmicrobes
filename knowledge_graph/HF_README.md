---
license: cc-by-4.0
task_categories:
  - graph-ml
tags:
  - knowledge-graph
  - microbiome
  - soil
  - metagenomics
pretty_name: SoilFM2 Knowledge Graph
---

# SoilFM2 Knowledge Graph

Integrated knowledge graph for soil microbiome research, combining microbial taxonomy, metabolic pathways, and metabolite information.

## Contents

| File | Description | Size |
|------|-------------|------|
| `soilfm_integrated_kg.pt` | PyTorch geometric graph with microbe/pathway/metabolite edges | ~71 MB |
| `integrated_node_mappings.pkl` | Node name → index mappings | ~54 MB |

## Statistics

- **Total nodes**: 2.39M
- **Node types**: Microbes, Pathways, Metabolites
- **Edge types**: Microbe-Pathway, Pathway-Metabolite associations

## Usage

```python
import torch
import pickle

# Load graph
kg = torch.load('soilfm_integrated_kg.pt')

# Load mappings
with open('integrated_node_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
```

## Data Sources

This knowledge graph integrates data from two sources:

### KG-Microbe

- **Repository**: https://github.com/Knowledge-Graph-Hub/kg-microbe
- **Citation**: Knowledge Graph Hub project

### MetagenomicKG

- **Repository**: https://github.com/KoslickiLab/MetagenomicKG
- **Publication**: Ma, C., Liu, S., & Koslicki, D. (2024). MetagenomicKG: A Knowledge Graph for Metagenomic Applications. *Briefings in Bioinformatics*, Volume 25, Issue 3, bbae119. https://doi.org/10.1093/bib/bbae119

## Citation

If you use this knowledge graph, please cite the original data sources above and:

```
Northen Lab, Lawrence Berkeley National Laboratory
SoilFM2: Multi-modal Foundation Model for Soil Microbiome
```

## License

CC-BY-4.0 (following the licenses of the source knowledge graphs)
