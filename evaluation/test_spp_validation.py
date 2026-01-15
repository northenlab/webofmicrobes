#!/usr/bin/env python3
"""
Example script demonstrating how to use the genus-average baseline model.

This script shows the same interface as the GNN-based model, making it easy
to compare baseline performance against more sophisticated approaches.
"""

from baseline_genus_average import GenusAverageBaseline
from pathlib import Path


def main():
    """
    Load and test the genus-average baseline model.
    """
    print("=== Genus-Average Baseline Model Demo ===\n")

    # Load pre-trained model
    model = GenusAverageBaseline()
    profiles_path = Path(__file__).parent / "genus_profiles.json"
    model.load_profiles(str(profiles_path))

    # Test organisms from different sources
    test_organisms = [
        ("BacillusOAE603", "17-member SynCom"),
        ("ArthrobacterOAP107", "17-member SynCom"),
        ("VariovoraxOAS795", "17-member SynCom"),
        ("FW507-14D01", "Biggs dataset"),
        ("GW101-11A03", "Biggs dataset - Acidovorax"),
        ("FW305-3-2-15-E-R2A2", "Biggs dataset - Pedobacter"),
        ("SomeUnknownBugXYZ", "Not in training"),
    ]

    print("\n" + "=" * 80)
    for organism, source in test_organisms:
        print(f"\n{'='*80}")
        print(f"Organism: {organism}")
        print(f"Source: {source}")
        genus = model.extract_genus(organism)
        print(f"Extracted genus: {genus}")

        # Check if genus is in training
        if genus in model.genus_profiles:
            print(f"Status: Using genus-specific profile ('{genus}')")
        else:
            print(f"Status: Using population mean (genus '{genus}' not in training)")

        # Get top consumed metabolites
        print(f"\nTop 10 Consumed Metabolites (substrate preferences):")
        top_consumed = model.get_top_consumed(organism, n=10)
        for i, (metabolite, score) in enumerate(top_consumed, 1):
            print(f"  {i:2d}. {metabolite:60s} {score:6.3f}")

        # Get top produced metabolites
        print(f"\nTop 10 Produced Metabolites:")
        top_produced = model.get_top_produced(organism, n=10)
        for i, (metabolite, score) in enumerate(top_produced, 1):
            print(f"  {i:2d}. {metabolite:60s} {score:6.3f}")

    print("\n" + "=" * 80)
    print("\n=== Model Statistics ===")
    print(f"Total genera in model: {len(model.genus_profiles)}")
    print(f"Total metabolites tracked: {len(model.metabolites)}")
    print(f"Organism mappings available: {len(model.organism_to_genus)}")

    print("\nGenera available in model:")
    for genus in sorted(model.genus_profiles.keys()):
        if genus != "Unknown":
            print(f"  - {genus}")

    print("\n=== Usage Notes ===")
    print("""
This baseline model provides a simple, interpretable reference for comparing
more sophisticated models. Key characteristics:

1. **Strengths**:
   - Fast inference (simple lookup)
   - Interpretable predictions
   - Reasonable for closely related organisms
   - Good baseline for genus-level patterns

2. **Limitations**:
   - No species-level resolution
   - Cannot learn from individual organism variation
   - No transfer learning across genera
   - Falls back to population mean for unknown genera

3. **Use cases**:
   - Baseline comparison for ML models
   - Quick predictions for well-studied genera
   - Understanding genus-level metabolic patterns
   - Identifying unusual organism behavior (deviations from genus average)

4. **Interface compatibility**:
   This model provides the same predict() and get_top_consumed() interface
   as GNNPreferenceHeadContrastive, making it easy to swap in different
   models for comparison.
    """)


if __name__ == "__main__":
    main()
