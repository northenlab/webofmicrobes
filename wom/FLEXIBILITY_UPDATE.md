# Query Flexibility Update

## Summary

Added consistent options across all prediction commands for maximum flexibility.

## New Global Options

### --explain (All prediction commands)
Shows detailed computation steps including:
- Embedding source (isolate vs aggregated)
- Number of isolates averaged
- Prediction method  
- Score statistics
- Background comparisons

**Available on:** `compounds`, `microbes`, `selective`, `fitness`, `niche`, `competition`, `rank-substrates`

### --use-distinctive (Complex queries)
Use distinctive scores instead of raw scores for:
- **selective**: Compare distinctive preferences (not just raw scores)
- **fitness**: Rank by distinctive fitness (unique advantage in environment)
- **competition**: Compete based on distinctive scores
- **rank-substrates**: Rank by distinctive preferences

**Default**: Raw scores (for backward compatibility)

### --agg (Aggregation options)
Control how scores are aggregated:
- **microbes**: `mean`, `sum`, `max` (default: mean)
- **fitness**: `sum`, `mean`, `weighted` (default: sum)
- **rank-substrates**: `mean`, `median`, `min`, `max` (optional, for multi-microbe aggregation)

## Updated Commands

### compounds
```bash
# Already had these options:
--explain           # NEW: Now consistent with other commands
--no-distinctive    # Disable distinctive scores
--background        # Background level for comparison
--min-score         # Threshold
```

### microbes  
```bash
# NEW OPTIONS:
--explain           # Show computation details
--agg mean|sum|max  # How to aggregate across compounds (default: mean)
```

### selective
```bash
# NEW OPTIONS:
--use-distinctive   # Use distinctive scores instead of raw
--explain           # Show computation details
--method            # Already existed: difference vs ratio
```

### fitness
```bash
# NEW OPTIONS:
--use-distinctive   # Use distinctive fitness scores
--explain           # Show computation details
--method            # Already existed: sum, mean, weighted
```

### niche
```bash
# NEW OPTIONS:
--explain           # Show computation details
# (Already always uses distinctive scores by definition)
```

### competition
```bash
# NEW OPTIONS:
--use-distinctive   # Compete based on distinctive scores
--explain           # Show computation details
```

### rank-substrates
```bash
# NEW OPTIONS:
--use-distinctive   # Rank by distinctive scores
--agg               # Aggregate across microbes (if desired)
--explain           # Show computation details
```

## Use Cases

### Example 1: Median aggregation
```bash
# Which microbes consume glucose OR glutamate? (median preference)
python scripts/wom.py --model-dir trained_models/bpr_consumed microbes \
  --compounds "glucose,glutamate" \
  --level genus \
  --agg median
```

### Example 2: Distinctive competition
```bash
# Who has UNIQUE advantage for these metabolites?
python scripts/wom.py --model-dir trained_models/bpr_consumed competition \
  --microbes "genus:Pseudomonas,genus:Burkholderia" \
  --metabolites "glucose,glutamate" \
  --use-distinctive
```

### Example 3: Explain everything
```bash
# Show me exactly how this works
python scripts/wom.py --model-dir trained_models/bpr_consumed selective \
  --target genus:Pseudomonas \
  --competitors genus:Burkholderia \
  --use-distinctive \
  --explain
```

### Example 4: Distinctive fitness
```bash
# Which genera have UNIQUE fitness (not just high raw scores)?
python scripts/wom.py --model-dir trained_models/bpr_consumed fitness \
  --metabolites "arginine,lysine,glutamate" \
  --level genus \
  --use-distinctive \
  --explain
```

## Implementation Status

✅ **CLI flags added** to all commands
✅ **Dataclasses updated** with new fields
✅ **CLI wiring complete** - args passed to queries
⚠️ **Query engine implementation** - Needs update to respect these flags

## Next Steps

Query engine handlers need to:
1. Check `query.explain` and add explanation dict to results
2. Check `query.use_distinctive` and use distinctive scores when True
3. Support `median` aggregation in addition to mean/sum/max
4. Pass explain info through to all result objects

## Benefits

- **Consistency**: Same options work across all commands
- **Flexibility**: Users can choose raw vs distinctive, mean vs median
- **Traceability**: --explain shows exactly what's happening
- **No breaking changes**: All options are additive with sensible defaults
