#!/usr/bin/env python
"""
Generate tool documentation for the system prompt by introspecting the actual code.

This ensures the system prompt always matches the implementation.
"""

import sys
from pathlib import Path
import inspect

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.query_types import (
    CompoundsForMicrobesQuery,
    MicrobesForCompoundsQuery,
    SelectiveMetabolitesQuery,
    MetaboliteEnvironmentFitnessQuery,
    NicheMetabolitesQuery,
    CompetitionAnalysisQuery,
    RankSubstratesQuery,
    SearchCompoundQuery,
    ListTaxaQuery,
    ModelInfoQuery,
)


COMMAND_METADATA = {
    "compounds": {
        "title": "What compounds does this microbe consume?",
        "when_to_use": 'User asks "what does Burkholderia eat", "what substrates does this genus prefer", "compound preferences for X"',
        "query_class": CompoundsForMicrobesQuery,
        "example": {
            "tool": "wom",
            "command": "compounds",
            "microbes": "genus:Pseudomonas",
            "top": 200,
            "explain": True
        },
        "returns": "Metabolites ranked by consumption preference for the specified microbe(s)"
    },
    "microbes": {
        "title": "What microbes consume this compound?",
        "when_to_use": 'User asks "which microbes eat X", "who consumes glutamate", "what organisms prefer this substrate"',
        "query_class": MicrobesForCompoundsQuery,
        "example": {
            "tool": "wom",
            "command": "microbes",
            "compounds": "glucose",
            "level": "genus",
            "top": 200,
            "agg": "mean",
            "explain": True
        },
        "returns": "Ranked list of microbes that consume the compound(s), with aggregate scores and validation results"
    },
    "selective": {
        "title": "Which metabolites favor one microbe over others?",
        "when_to_use": '"What substrates give Burkholderia an advantage over Pseudomonas?", "Which metabolites are selective for X over Y?"',
        "query_class": SelectiveMetabolitesQuery,
        "example": {
            "tool": "wom",
            "command": "selective",
            "target": "genus:Pseudomonas",
            "competitors": "genus:Burkholderia,genus:Variovorax",
            "top": 200,
            "method": "difference",
            "use_distinctive": False,
            "explain": True
        },
        "returns": "Metabolites ranked by selectivity (positive = target prefers more than competitors)"
    },
    "fitness": {
        "title": "Which microbes thrive in this metabolite environment?",
        "when_to_use": '"Which microbes grow well with these metabolites available?", "What organisms would dominate with [compounds]?"',
        "query_class": MetaboliteEnvironmentFitnessQuery,
        "example": {
            "tool": "wom",
            "command": "fitness",
            "metabolites": "glutamate,arginine,lysine",
            "level": "genus",
            "top": 200,
            "fitness_method": "sum",
            "use_distinctive": False,
            "explain": True
        },
        "returns": "Microbes ranked by fitness in the given metabolite environment"
    },
    "niche": {
        "title": "What's metabolically unique about this microbe?",
        "when_to_use": '"What makes Pseudomonas special metabolically?", "Unique preferences of this organism"',
        "query_class": NicheMetabolitesQuery,
        "example": {
            "tool": "wom",
            "command": "niche",
            "microbe": "genus:Pseudomonas",
            "top": 200,
            "explain": True
        },
        "returns": "Metabolites with distinctive scores showing what's unique to this microbe"
    },
    "competition": {
        "title": "Who wins between microbes for specific metabolites?",
        "when_to_use": '"Does Burkholderia or Pseudomonas prefer glucose?", "Metabolic competitive advantage"',
        "query_class": CompetitionAnalysisQuery,
        "example": {
            "tool": "wom",
            "command": "competition",
            "microbes": "genus:Burkholderia,genus:Pseudomonas",
            "metabolites": "glucose,glutamate",
            "use_distinctive": False,
            "explain": True
        },
        "returns": "Winner for each metabolite, margin of victory, all competitor scores, aggregated win counts"
    },
    "rank-substrates": {
        "title": "Rank these substrates for these microbes",
        "when_to_use": '"How do these microbes rank these compounds?", "Substrate preference ranking"',
        "query_class": RankSubstratesQuery,
        "example": {
            "tool": "wom",
            "command": "rank-substrates",
            "microbes": "genus:Bacillus,genus:Pseudomonas",
            "substrates": "glucose,glutamate,alanine",
            "top": 200,
            "use_distinctive": False,
            "explain": True
        },
        "returns": "For each microbe, a ranked list of substrates by preference"
    },
    "search": {
        "title": "Search for a compound or metabolite",
        "when_to_use": "User unsure of exact compound name, needs to find alternatives, fuzzy matching",
        "query_class": SearchCompoundQuery,
        "example": {
            "tool": "wom",
            "command": "search",
            "query": "glutamic",
            "fuzzy": True,
            "min_score": 0.8
        },
        "returns": "List of matching compounds with similarity scores"
    },
    "list": {
        "title": "List all available organisms at a level",
        "when_to_use": '"What genera are in the model?", "Show all species", "Available taxa"',
        "query_class": ListTaxaQuery,
        "example": {
            "tool": "wom",
            "command": "list",
            "level": "genus"
        },
        "returns": "List of all taxa at the specified level with counts"
    },
    "info": {
        "title": "Model information",
        "when_to_use": '"How many microbes in the model?", "What compounds are available?", "Model statistics"',
        "query_class": ModelInfoQuery,
        "example": {
            "tool": "wom",
            "command": "info"
        },
        "returns": "Model metadata, statistics, training information"
    }
}


def extract_parameters(query_class):
    """Extract parameter information from a query class."""
    # Get the from_string or from_strings method signature
    if hasattr(query_class, 'from_strings'):
        method = query_class.from_strings
    elif hasattr(query_class, 'from_string'):
        method = query_class.from_string
    else:
        # For simple query classes like SearchCompoundQuery
        method = query_class.__init__
    
    sig = inspect.signature(method)
    params = {}
    
    for name, param in sig.parameters.items():
        if name in ('cls', 'self', 'config'):
            continue
        
        param_info = {
            "name": name,
            "required": param.default == inspect.Parameter.empty,
            "default": None if param.default == inspect.Parameter.empty else param.default,
            "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"
        }
        params[name] = param_info
    
    return params


def generate_command_docs():
    """Generate markdown documentation for all commands."""
    
    docs = """## COMMAND REFERENCE - COMPLETE GUIDE

*This documentation is auto-generated from the codebase to ensure accuracy.*

### A. BASIC QUERIES (High-Level Substrate Exploration)

"""
    
    # Basic queries
    basic_commands = ["microbes", "compounds"]
    for i, cmd in enumerate(basic_commands, 1):
        meta = COMMAND_METADATA[cmd]
        docs += f"""#### Query {i}: "{meta['title']}"
**Command**: `{cmd}`
**When to use**: {meta['when_to_use']}

**Tool Format**:
```json
{format_json(meta['example'])}
```

**Returns**: {meta['returns']}

"""
    
    # Comparative analysis
    docs += """---

### B. COMPARATIVE ANALYSIS (Understanding Microbial Relationships)

"""
    
    comparative_commands = ["selective", "fitness", "niche", "competition", "rank-substrates"]
    for i, cmd in enumerate(comparative_commands, 3):
        meta = COMMAND_METADATA[cmd]
        docs += f"""#### Query {i}: "{meta['title']}"
**Command**: `{cmd}`
**When to use**: {meta['when_to_use']}

**Tool Format**:
```json
{format_json(meta['example'])}
```

**Returns**: {meta['returns']}

"""
    
    # Utility queries
    docs += """---

### C. UTILITY QUERIES (Exploration & Discovery)

"""
    
    utility_commands = ["search", "list", "info"]
    for i, cmd in enumerate(utility_commands, 8):
        meta = COMMAND_METADATA[cmd]
        docs += f"""#### Query {i}: "{meta['title']}"
**Command**: `{cmd}`
**When to use**: {meta['when_to_use']}

**Tool Format**:
```json
{format_json(meta['example'])}
```

**Returns**: {meta['returns']}

"""
    
    # Growth rate tool
    docs += """---

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

"""
    
    return docs


def format_json(obj, indent=2):
    """Format a JSON object with proper indentation."""
    import json
    return json.dumps(obj, indent=indent)


def main():
    """Generate and print command documentation."""
    docs = generate_command_docs()
    print(docs)
    
    # Optionally save to file
    output_path = Path(__file__).parent.parent / "docs" / "command_reference_generated.md"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(docs)
    print(f"\n✅ Documentation saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
