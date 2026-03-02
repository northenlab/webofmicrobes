# Command Reference

Complete reference for all `wom.py` commands.

## Basic Queries

### compounds
**What metabolites does this microbe consume/produce?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed compounds \
  --microbes "genus:Burkholderia" \
  --top 20 \
  --min-score 0.5 \
  --explain
```

Options:
- `--microbes`: Microbe query (e.g., `genus:Bosea`, `isolate:XYZ`)
- `--top`: Number of results (default: 20)
- `--min-score`: Minimum score threshold
- `--no-distinctive`: Show only raw scores
- `--background`: Background level for distinctive scores
- `--explain`: Show computation details

### microbes
**Which microbes consume/produce these metabolites?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed microbes \
  --compounds "glucose,pyruvate,alanine" \
  --level genus \
  --top 10 \
  --agg sum
```

Options:
- `--compounds`: Comma-separated compound names
- `--level`: Taxonomic level (genus, family, order, class)
- `--top`: Number of results (default: 10)
- `--agg`: Aggregation method (mean, sum, max)
- `--min-score`: Minimum score threshold

## Complex Queries

### selective
**Which metabolites favor the target microbe over competitors?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed selective \
  --target "genus:Pseudomonas" \
  --competitors "genus:Burkholderia,genus:Variovorax" \
  --top 10 \
  --method difference
```

Use case: *"Which metabolite should I add to promote Pseudomonas over Burkholderia?"*

Options:
- `--target`: Target microbe to promote
- `--competitors`: Competitor microbes (comma-separated)
- `--top`: Number of results
- `--method`: Selection method (difference, ratio)
- `--min-selectivity`: Minimum selectivity score

### fitness
**Which microbes thrive in this metabolite environment?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed fitness \
  --metabolites "arginine,lysine,glutamate" \
  --level genus \
  --top 10 \
  --method sum
```

Use case: *"In an environment with arginine and lysine, which microbes grow best?"*

Options:
- `--metabolites`: Metabolites in environment (comma-separated)
- `--microbes`: Restrict to specific microbes (optional)
- `--level`: Taxonomic level
- `--top`: Number of results
- `--method`: Fitness method (sum, mean, max)

### niche
**What makes this microbe metabolically unique?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed niche \
  --microbe "genus:Pseudomonas" \
  --top 20 \
  --background genus
```

Use case: *"What metabolites does Pseudomonas uniquely prefer compared to other genera?"*

Options:
- `--microbe`: Target microbe
- `--top`: Number of results
- `--background`: Comparison level (genus, family, etc.)
- `--min-distinctive`: Minimum distinctive score

### competition
**Who wins in competition for specific metabolites?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed competition \
  --microbes "genus:Pseudomonas,genus:Burkholderia" \
  --metabolites "glucose,glutamate"
```

Use case: *"If Pseudomonas and Burkholderia compete for glucose and glutamate, who has the advantage?"*

Output:
```json
{
  "results": [
    {
      "metabolite": "glucose",
      "winner": "genus:Burkholderia",
      "winner_score": 4.6886,
      "all_scores": {
        "genus:Pseudomonas": 4.5306,
        "genus:Burkholderia": 4.6886
      },
      "margin": 0.158
    }
  ],
  "metadata": {
    "win_counts": {
      "genus:Pseudomonas": 1,
      "genus:Burkholderia": 1
    },
    "overall_winner": "genus:Pseudomonas"
  }
}
```

Options:
- `--microbes`: Competing microbes (comma-separated)
- `--metabolites`: Metabolites to compete for (comma-separated)

### rank-substrates
**How does each microbe rank these specific substrates?**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed rank-substrates \
  --microbes "genus:Bacillus,genus:Pseudomonas" \
  --substrates "glucose,glutamate,alanine" \
  --top 10
```

Use case: *"For Bacillus and Pseudomonas, rank these substrates by preference."*

Output:
```json
{
  "results": [
    {
      "microbe": {"level": "genus", "name": "Bacillus"},
      "rankings": [
        {"substrate": "alanine", "matched": "alanine", "score": 3.6372},
        {"substrate": "glucose", "matched": "glucose", "score": 2.1973},
        {"substrate": "glutamate", "matched": "glutamate(1−)", "score": -4.7734}
      ]
    }
  ]
}
```

Options:
- `--microbes`: Microbes to analyze (comma-separated)
- `--substrates`: Substrates to rank (comma-separated)
- `--top`: Number of top substrates per microbe

## Utility Commands

### search
**Find compound name matches**

```bash
python scripts/wom.py search "glutamic acid"
```

Options:
- `--exact`: Exact match only (no fuzzy matching)
- `--min-score`: Minimum fuzzy match score (default: 0.7)

### list
**List available taxa**

```bash
python scripts/wom.py list --level genus
```

Options:
- `--level`: Taxonomic level (genus, family, order, class, kingdom)
- `--no-counts`: Don't include isolate counts

### info
**Get model information**

```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed info
```

Options:
- `--no-stats`: Exclude detailed statistics

## Global Options

All commands support:
- `--model-dir`: Path to trained model (required for most commands)
- `--format`: Output format (`json` or `table`)
- `--verbose`: Enable verbose logging
- `--explain`: Show computation details (select commands)

## Output Formats

### JSON (default)
```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed compounds --microbes genus:Bosea
```

### Table
```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed compounds --microbes genus:Bosea --format table
```

## Model Selection

Different models for different purposes:

```bash
# Consumed metabolites (decreased concentration)
--model-dir trained_models/bpr_consumed
--model-dir trained_models/wrmf_consumed
--model-dir trained_models/svd_consumed

# Secreted metabolites (increased concentration)
--model-dir trained_models/bpr_secreted
--model-dir trained_models/wrmf_secreted
--model-dir trained_models/svd_secreted
```

**Model characteristics:**
- **BPR**: Best overall performance (32 dimensions)
- **WR-MF**: Best for sparse data (128 dimensions)
- **SVD**: Fastest, good baseline (8 dimensions)
