# Algorithm Flow Documentation

## Query: "What does genus Bosea consume?"

### Command
```bash
wom compounds --microbes genus:Bosea --top 20
```

### What Happens Mechanically

#### Step 1: Model Loading (QueryEngine initialization)
```python
# api/query_engine.py, line 56-67
artifacts = load_model_artifacts(model_dir)
# Loads:
# - Model with trained embeddings (shape: [n_microbes, embedding_dim])
# - Training data with microbe/metabolite mappings
# - Taxonomy for each isolate

# THEN: Compute taxonomy embeddings
for level in ["genus", "family", ...]:
    genus_embeddings = compute_taxonomy_embeddings(artifacts, "genus")
    # This averages all isolate embeddings in each genus
```

**Key Point**: Genus embeddings are **pre-computed averages** of isolate embeddings.

For genus "Bosea":
- If 1 isolate: genus_emb = isolate_emb (average of 1)  
- If 5 isolates: genus_emb = mean([iso1_emb, iso2_emb, iso3_emb, iso4_emb, iso5_emb])

#### Step 2: Get Embedding
```python
# core/types.py, ModelArtifacts.get_embedding()
if level == "isolate":
    embedding = model.microbe_embeddings[isolate_idx]  # Direct lookup
elif level == "genus":
    embedding = taxonomy_embeddings["genus"]["Bosea"]  # Pre-computed average
```

**Answer to "genus with 1 isolate"**: Uses the pre-computed genus embedding (which equals the isolate embedding), NOT a direct isolate lookup.

#### Step 3: Predict Scores
```python
# core/models.py, BaseEmbeddingModel.predict_scores_from_embedding()
scores = microbe_embedding @ metabolite_embeddings.T  # Dot product
# Shape: (embedding_dim,) @ (n_metabolites, embedding_dim).T
# Result: (n_metabolites,) - one score per metabolite
```

**The Core Algorithm**: Score for metabolite M = dot product of microbe embedding and M's embedding.

Higher score → microbe embedding is more aligned with metabolite embedding → higher predicted consumption.

#### Step 4: Compute Distinctive Scores (optional)
```python
# core/types.py, ModelArtifacts.predict_with_stats()
if compute_distinctive:
    # Get all genus embeddings as background
    background_embeddings = taxonomy_embeddings["genus"].values()
    
    # Predict for each background genus
    background_scores = [bg_emb @ metabolite_embeddings.T for bg_emb in background_embeddings]
    
    # Average across all genera
    background_mean = np.mean(background_scores, axis=0)
    
    # Distinctive = how much Bosea differs from average genus
    distinctive_scores = bosea_scores - background_mean
```

**Distinctive Score**: Metabolite's score for Bosea MINUS average score across all genera.
- Positive = Bosea scores higher than average genus
- Negative = Bosea scores lower than average genus

#### Step 5: Rank and Return
```python
# Sort by scores (raw or distinctive)
top_indices = np.argsort(scores)[-20:][::-1]  # Top 20

# Add statistics
mean_score = np.mean(scores)
median_score = np.median(scores)
std_score = np.std(scores)
ranks = np.argsort(np.argsort(-scores)) + 1  # 1-indexed ranks
percentiles = (ranks / n_metabolites) * 100
```

## Complete Call Stack

```
wom.py: cmd_compounds()
  ↓
QueryEngine.query_compounds_for_microbes(query)
  ↓
ModelArtifacts.predict_with_stats(level="genus", name="Bosea")
  ↓
ModelArtifacts.get_embedding(level="genus", name="Bosea")
  → Returns: taxonomy_embeddings["genus"]["Bosea"]  (pre-computed average)
  ↓
BaseEmbeddingModel.predict_scores_from_embedding(embedding)
  → Returns: embedding @ metabolite_embeddings.T  (DOT PRODUCT)
  ↓
Compute distinctive scores (optional):
  → bosea_scores - mean(all_genus_scores)
  ↓
Rank and add statistics
```

## Key Insights

### 1. Genus Embeddings are Always Averages
Even if genus has 1 isolate, it uses the pre-computed genus embedding (= that isolate's embedding).

### 2. Isolate vs Genus Lookup
- `--microbes isolate:XYZ`: Direct lookup in model.microbe_embeddings
- `--microbes genus:XYZ`: Lookup in pre-computed taxonomy_embeddings["genus"]
- These are **different lookups**, even if genus has 1 isolate

### 3. The Core Prediction
**Score = dot(microbe_embedding, metabolite_embedding)**

This is the fundamental operation. Everything else is aggregation or statistics.

### 4. Distinctive Scores
**Distinctive = my_score - average_score_across_peers**

Shows what's unique about this microbe compared to its taxonomic peers.

## FAQ

**Q: If genus has 1 isolate, is genus embedding identical to isolate embedding?**  
A: Yes, mathematically identical (mean of 1 value = that value). But different lookup paths.

**Q: Does --reduce mean do anything for single-isolate genera?**  
A: The --reduce parameter isn't shown in these examples. If it existed, it would only matter when aggregating across MULTIPLE microbes in a query, not within a single genus.

**Q: How does the model learn these embeddings?**  
A: BPR training optimizes: "isolates that consume metabolite M should have embeddings close to M's embedding" (via dot product). Trained on pairwise comparisons: push positive pairs together, negative pairs apart.

**Q: Why pre-compute genus embeddings instead of averaging on-the-fly?**  
A: Performance. Computing on-the-fly would require averaging isolate embeddings for every query. Pre-computing does it once.

**Q: Can I see the actual embedding values?**  
A: Yes, add --verbose flag (or implement explain mode) to show:
- Embedding vector
- Which isolates were averaged (for taxa)
- Raw score distribution
- Background statistics
