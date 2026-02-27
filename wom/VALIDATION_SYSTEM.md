# Input Validation System Documentation

## Overview

The web-of-microbes system now includes a comprehensive input validation framework that addresses critical real-world usability issues discovered during beta testing. When users submit long lists of compounds or taxonomic groups, the system now:

1. **Validates every input** before processing
2. **Reports what matched and what didn't** with detailed reasons
3. **Fails fast** when nothing matches (no confusing empty results)
4. **Warns prominently** when many items don't match
5. **Shows matching details** in `--explain` mode

## Problem Statement

During real-world beta testing, users reported confusion when:
- They submitted long lists of compounds with typos (e.g., "froctose" instead of "fructose")
- Some of their taxonomic groups weren't in the model
- Results didn't match their expectations because failed inputs were silently ignored
- No feedback indicated which items matched vs which didn't

## Solution Architecture

### Core Components

#### 1. `InputMatchResult` (core/validation.py)
Tracks validation status for a single input:

```python
@dataclass
class InputMatchResult:
    requested: str                          # What user typed
    matched: Optional[str]                  # What we used (None if failed)
    match_type: str                         # 'exact', 'fuzzy', 'not_found', 'not_in_model'
    match_score: float                      # 0.0 to 1.0 for fuzzy matches
    reason: Optional[str]                   # Why it didn't match
    
    @property
    def success(self) -> bool:
        return self.matched is not None and self.match_type != 'not_in_model'
```

**Match Types:**
- `exact`: Matched exactly in ChEBI database and is in model
- `fuzzy`: Fuzzy-matched in ChEBI but not exact match (rare, typically found via fuzzy search)
- `not_found`: No match in ChEBI database (typo or completely wrong name)
- `not_in_model`: Matched in ChEBI but not in this model's training data

#### 2. `InputValidationSummary` (core/validation.py)
Aggregates validation results and computes warnings:

```python
@dataclass
class InputValidationSummary:
    total_requested: int                    # User submitted this many inputs
    total_matched: int                      # System found this many
    total_failed: int                       # System couldn't match this many
    match_rate: float                       # matched / requested
    matched_items: list[InputMatchResult]   # Successful matches
    failed_items: list[InputMatchResult]    # Failed matches
    warnings: list[str]                     # Auto-generated warnings
    should_warn: bool                       # match_rate < 0.8
    should_fail: bool                       # match_rate == 0.0
```

**Thresholds:**
- `should_warn = True` when match_rate < 80%: User should review what matched
- `should_fail = True` when match_rate == 0%: Query cannot proceed, fail fast

#### 3. Validation Helpers in QueryEngine (api/query_engine.py)

```python
def validate_metabolites(names: list[str]) -> InputValidationSummary
    # Check each metabolite via ChEBI compound_mapper
    # Verify it's in training data
    # Return full validation summary with reasons
    
def validate_taxa(taxa_queries: list[tuple[str, str]]) -> InputValidationSummary
    # Check each taxon (level, name) exists at that level
    # Handle special case of isolates
    # Return validation summary
```

### Integration Pattern

Each query handler follows this pattern:

```python
def query_something(self, query: SomeQuery) -> QueryResponse:
    try:
        # Step 1: VALIDATE INPUTS UPFRONT
        validation = self.validate_metabolites(query.compounds)
        
        # Step 2: FAIL FAST IF NOTHING MATCHED
        if validation.should_fail:
            return QueryResponse(
                success=False,
                error=f"None of the {validation.total_requested} items matched",
                metadata={"input_validation": validation.to_dict()},
            )
        
        # Step 3: PROCESS ONLY MATCHED ITEMS
        for match_result in validation.matched_items:
            # ... use match_result.matched instead of original query input
            
            if query.explain:
                # Show what was matched
                explanation["input_matching"] = {
                    "requested": match_result.requested,
                    "matched": match_result.matched,
                    "match_type": match_result.match_type,
                }
        
        # Step 4: INCLUDE VALIDATION IN RESPONSE
        metadata["input_validation"] = validation.to_dict()
        if validation.should_warn:
            metadata["warnings"] = validation.warnings
        
        return QueryResponse(success=True, metadata=metadata, ...)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return QueryResponse(success=False, error=str(e))
```

## User-Facing Behavior

### Example 1: Mixed exact, fuzzy, and failed matches

```bash
$ wom microbes --compounds "glucose,froctose,arabinose,NOTAREALCOMPOUND"

⚠️  INPUT VALIDATION WARNINGS:
   3/4 inputs did not match: "froctose", "arabinose", "NOTAREALCOMPOUND"
   See metadata.input_validation for details

{
  "success": true,
  "results": [
    { "taxon": "Escherichia", "score": 8.93, ... },
    ...
  ],
  "metadata": {
    "input_validation": {
      "match_rate": 0.25,
      "matched_items": [
        {"requested": "glucose", "matched": "glucose", "match_type": "exact"}
      ],
      "failed_items": [
        {"requested": "froctose", "reason": "No match found in ChEBI database"},
        {"requested": "arabinose", "matched": "arabinose", "match_type": "not_in_model"},
        {"requested": "NOTAREALCOMPOUND", "reason": "No match found in ChEBI database"}
      ]
    }
  }
}
```

### Example 2: Fail-fast when nothing matches

```bash
$ wom microbes --compounds "FAKE1,FAKE2,FAKE3"

{
  "success": false,
  "results": [],
  "error": "None of the 3 compounds could be matched or are in the model",
  "metadata": {
    "input_validation": {
      "match_rate": 0.0,
      "total_requested": 3,
      "total_matched": 0,
      "failed_items": [
        {"requested": "FAKE1", "reason": "No match found in ChEBI database"},
        {"requested": "FAKE2", "reason": "No match found in ChEBI database"},
        {"requested": "FAKE3", "reason": "No match found in ChEBI database"}
      ]
    }
  }
}
```

### Example 3: With --explain flag

```bash
$ wom microbes --compounds "glucose,fructose" --explain

{
  "results": [
    {
      "taxon": "Escherichia",
      "compound_scores": {
        "glucose": 7.23,
        ...
      },
      "explanation": {
        "input_matching": {
          "requested": "glucose",
          "matched": "glucose",
          "match_type": "exact"
        },
        ...
      }
    }
  ],
  "metadata": {
    "input_validation": {
      "matched_items": [
        {
          "requested": "glucose",
          "matched": "glucose",
          "match_type": "exact",
          "match_score": 1.0
        }
      ],
      "failed_items": [
        {
          "requested": "fructose",
          "matched": "β-D-fructofuranose",
          "match_type": "not_in_model",
          "match_score": 1.0,
          "reason": "Matched 'β-D-fructofuranose' but not in training data"
        }
      ]
    }
  }
}
```

## Applied Query Handlers

The validation framework is integrated into all 6 data query handlers:

1. **microbes_for_compounds**: Validates metabolite list
2. **selective_metabolites**: Validates target microbe + competitors
3. **metabolite_environment_fitness**: Validates metabolites + optional microbes
4. **niche_metabolites**: Validates single microbe
5. **competition_analysis**: Validates microbes + metabolites
6. **rank_substrates**: Validates microbes + substrates

Utility queries (search, list, info) don't use validation as they have different purposes.

## Implementation Files

- **core/validation.py**: Core validation framework (InputMatchResult, InputValidationSummary)
- **api/query_engine.py**: 
  - validate_metabolites() helper
  - validate_taxa() helper
  - Updated all 6 query handlers
- **scripts/wom.py**: Enhanced output_response() to show warnings prominently
- **api/query_types.py**: All query types have explain field for traceability

## Testing

Comprehensive test scenarios implemented:

1. ✅ 100% exact matches (baseline)
2. ✅ Mix of exact and fuzzy matches
3. ✅ 50% match rate (warning threshold)
4. ✅ 0% match rate (fail-fast)
5. ✅ ChEBI matches not in training data
6. ✅ Completely fake inputs
7. ✅ Typos in compound names
8. ✅ Missing taxonomic groups
9. ✅ Mixed valid/invalid microbes and metabolites

## Benefits

1. **Transparency**: Users see exactly what matched and why each item failed
2. **Fail-Fast**: No more confusing empty results - clear error messages upfront
3. **Guidance**: Warnings prompt users to review unexpected failures
4. **Traceability**: --explain mode shows full matching details
5. **Consistency**: All 6 data queries use the same validation framework
6. **Beta Tester Friendly**: Addresses the exact problem reported in real-world testing

## Future Enhancements

1. Fuzzy matching for taxonomic names (currently only exact matches)
2. Suggested corrections for typos ("Did you mean...")
3. Batch processing statistics (X matched, Y failed, average confidence)
4. Interactive mode to let users resolve ambiguities
5. Match confidence scoring for fuzzy metabolite matches
