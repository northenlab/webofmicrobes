## COMMAND REFERENCE - COMPLETE GUIDE

*This documentation is auto-generated from the codebase to ensure accuracy.*

### A. BASIC QUERIES (High-Level Substrate Exploration)

#### Query 1: "What microbes consume this compound?"
**Command**: `microbes`
**When to use**: User asks "which microbes eat X", "who consumes glutamate", "what organisms prefer this substrate"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "microbes",
  "compounds": "glucose",
  "level": "genus",
  "top": 10,
  "agg": "mean",
  "explain": false
}
```

**Returns**: Ranked list of microbes that consume the compound(s), with aggregate scores and validation results

#### Query 2: "What compounds does this microbe consume?"
**Command**: `compounds`
**When to use**: User asks "what does Burkholderia eat", "what substrates does this genus prefer", "compound preferences for X"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "compounds",
  "microbes": "genus:Pseudomonas",
  "top": 20,
  "explain": false
}
```

**Returns**: Metabolites ranked by consumption preference for the specified microbe(s)

---

### B. COMPARATIVE ANALYSIS (Understanding Microbial Relationships)

#### Query 3: "Which metabolites favor one microbe over others?"
**Command**: `selective`
**When to use**: "What substrates give Burkholderia an advantage over Pseudomonas?", "Which metabolites are selective for X over Y?"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "selective",
  "target": "genus:Pseudomonas",
  "competitors": "genus:Burkholderia,genus:Variovorax",
  "top": 10,
  "method": "difference",
  "use_distinctive": false,
  "explain": false
}
```

**Returns**: Metabolites ranked by selectivity (positive = target prefers more than competitors)

#### Query 4: "Which microbes thrive in this metabolite environment?"
**Command**: `fitness`
**When to use**: "Which microbes grow well with these metabolites available?", "What organisms would dominate with [compounds]?"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "fitness",
  "metabolites": "glutamate,arginine,lysine",
  "level": "genus",
  "top": 10,
  "fitness_method": "sum",
  "use_distinctive": false,
  "explain": false
}
```

**Returns**: Microbes ranked by fitness in the given metabolite environment

#### Query 5: "What's metabolically unique about this microbe?"
**Command**: `niche`
**When to use**: "What makes Pseudomonas special metabolically?", "Unique preferences of this organism"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "niche",
  "microbe": "genus:Pseudomonas",
  "top": 20,
  "explain": false
}
```

**Returns**: Metabolites with distinctive scores showing what's unique to this microbe

#### Query 6: "Who wins between microbes for specific metabolites?"
**Command**: `competition`
**When to use**: "Does Burkholderia or Pseudomonas prefer glucose?", "Metabolic competitive advantage"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "competition",
  "microbes": "genus:Burkholderia,genus:Pseudomonas",
  "metabolites": "glucose,glutamate",
  "use_distinctive": false,
  "explain": false
}
```

**Returns**: Winner for each metabolite, margin of victory, all competitor scores, aggregated win counts

#### Query 7: "Rank these substrates for these microbes"
**Command**: `rank-substrates`
**When to use**: "How do these microbes rank these compounds?", "Substrate preference ranking"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "rank-substrates",
  "microbes": "genus:Bacillus,genus:Pseudomonas",
  "substrates": "glucose,glutamate,alanine",
  "top": 10,
  "use_distinctive": false,
  "explain": false
}
```

**Returns**: For each microbe, a ranked list of substrates by preference

---

### C. UTILITY QUERIES (Exploration & Discovery)

#### Query 8: "Search for a compound or metabolite"
**Command**: `search`
**When to use**: User unsure of exact compound name, needs to find alternatives, fuzzy matching

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "search",
  "query": "glutamic",
  "fuzzy": true,
  "min_score": 0.8
}
```

**Returns**: List of matching compounds with similarity scores

#### Query 9: "List all available organisms at a level"
**Command**: `list`
**When to use**: "What genera are in the model?", "Show all species", "Available taxa"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "list",
  "level": "genus"
}
```

**Returns**: List of all taxa at the specified level with counts

#### Query 10: "Model information"
**Command**: `info`
**When to use**: "How many microbes in the model?", "What compounds are available?", "Model statistics"

**Tool Format**:
```json
{
  "tool": "wom",
  "command": "info"
}
```

**Returns**: Model metadata, statistics, training information

---

### D. GROWTH RATE QUERIES (External Tool)

#### Query 11: "What is the growth rate/doubling time?"
**Command**: `growth_rate` (separate tool, not WoM)
**When to use**: "How fast does X grow?", "What's the doubling time of Pseudomonas?", "Growth rate of this organism?"

**Tool Format**:
```json
{
  "tool": "growth_rate",
  "query": "Pseudomonas",
  "level": "genus",
  "stat": "median"
}
```

**Parameters**:
- `query`: Organism name to look up
- `level`: Taxonomic level (genus, species, etc.)
- `stat`: Statistic to return (median, mean, min, max)

**Returns**: Predicted doubling time in hours from Phydon growth rate predictions

**Integration tip**: Growth rate queries can be interleaved with metabolite preference queries to provide comprehensive metabolic profiles. Example: "Pseudomonas has a median doubling time of X hours and shows strong preference for glutamate (score 7.32)."

