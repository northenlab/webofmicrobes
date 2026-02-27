# Query Migration Checklist

Comparison between old `wom_complex.py` and new `wom.py` refactored system.

## OLD SYSTEM (wom_complex.py)

| Command | Purpose | Status |
|---------|---------|--------|
| **selective** | Find metabolites favoring target over competitors | ✅ MIGRATED |
| **fitness** | Rank microbes by affinity for metabolite environment | ✅ MIGRATED |
| **niche** | Find unique metabolite preferences for a microbe | ✅ MIGRATED |
| **competition** | Analyze competition between microbes for metabolites | ✅ MIGRATED |
| **rank-substrates** | Rank specific substrates for specific microbes | ✅ MIGRATED |
| **info** | Show available commands and examples | ✅ MIGRATED |

## NEW SYSTEM (wom.py)

| Command | Purpose | Notes |
|---------|---------|-------|
| **compounds** | What does this microbe consume/produce? | NEW - replaces old `wom.py` |
| **microbes** | Who consumes/produces these compounds? | NEW - replaces old `wom.py` |
| **selective** | Find metabolites favoring target over competitors | MIGRATED from wom_complex.py |
| **fitness** | Rank microbes by affinity for environment | MIGRATED from wom_complex.py |
| **niche** | Find unique metabolite preferences | MIGRATED from wom_complex.py |
| **competition** | Analyze competition between microbes | MIGRATED from wom_complex.py |
| **rank-substrates** | Rank specific substrates for microbes | MIGRATED from wom_complex.py |
| **search** | Search for metabolites/microbes | NEW |
| **list** | List all organisms at taxonomic level | NEW |
| **info** | Show model and data statistics | MIGRATED from wom_complex.py |

## ✅ MIGRATION COMPLETE

All queries from the old system have been successfully migrated to the new refactored architecture.

### Example Usage Comparison

**OLD:**
```bash
python wom_complex.py competition \
  --microbes "genus:Pseudomonas,genus:Burkholderia" \
  --metabolites "glucose,glutamate"
```

**NEW:**
```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed competition \
  --microbes "genus:Pseudomonas,genus:Burkholderia" \
  --metabolites "glucose,glutamate"
```

**OLD:**
```bash
python wom_complex.py rank-substrates \
  --microbes "genus:Bacillus,genus:Pseudomonas" \
  --substrates "glucose,glutamate,alanine"
```

**NEW:**
```bash
python scripts/wom.py --model-dir trained_models/bpr_consumed rank-substrates \
  --microbes "genus:Bacillus,genus:Pseudomonas" \
  --substrates "glucose,glutamate,alanine"
```
