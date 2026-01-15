#!/usr/bin/env python3
"""
Genus-Average Baseline Model for Substrate Preference Prediction

A simple baseline that predicts metabolite consumption/production for an organism
based on the average profile of all organisms from the same genus in the training data.

This provides a reasonable baseline to compare against more sophisticated models,
as organisms within the same genus often share metabolic capabilities.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class GenusAverageBaseline:
    """
    Baseline model that predicts substrate preferences using genus-level averages.

    For each genus seen in training:
    - Computes average metabolite profile across all organisms of that genus
    - Stores profiles for prediction

    For prediction:
    - Extracts genus from organism name
    - Returns genus average if available
    - Falls back to population mean if genus not seen in training
    """

    def __init__(self):
        self.genus_profiles: Dict[str, Dict[str, float]] = {}
        self.population_mean: Dict[str, float] = {}
        self.metabolites: List[str] = []
        self.organism_to_genus: Dict[str, str] = {}  # Explicit mapping for known organisms
        self.trained = False

    def extract_genus(self, organism_name: str) -> str:
        """
        Extract genus from organism name.

        Examples:
            "Bacillus sp. (FW507)" -> "Bacillus"
            "BacillusOAE603" -> "Bacillus"
            "ArthrobacterOAP107" -> "Arthrobacter"
            "FW305-3" -> "Unknown" (code with no genus info)

        Strategy:
            1. Check if organism is in explicit mapping (from training data)
            2. Check if name starts with capital letter followed by lowercase (genus pattern)
            3. Extract first word or CamelCase prefix
            4. Return "Unknown" if no genus can be extracted
        """
        organism_name = organism_name.strip()

        # First, check explicit mapping
        if organism_name in self.organism_to_genus:
            genus = self.organism_to_genus[organism_name]
            if pd.notna(genus) and genus != "":
                return genus

        # Pattern 1: "GenusCodeNumber" (e.g., "BacillusOAE603")
        # Extract genus by finding where lowercase->uppercase transition occurs after initial capital
        match = re.match(r'^([A-Z][a-z]+)', organism_name)
        if match:
            return match.group(1)

        # Pattern 2: "Genus species (code)" (e.g., "Bacillus sp. (FW507)")
        if ' ' in organism_name:
            parts = organism_name.split()
            if parts[0][0].isupper():
                return parts[0]

        # Pattern 3: Code only (e.g., "FW305-3", "EB106-06-01-XG185")
        # Cannot extract genus from code alone
        return "Unknown"

    def parse_assertion_code(self, assertion_code: str, log2fc: float) -> float:
        """
        Parse assertion code and log2FC into numerical score.

        Rules:
            - 'ND' = not detected -> NaN (excluded from averaging)
            - 'D' or empty = detected not significant -> 0.0
            - Numeric (-5 to +5) = use log2FC value

        For consumption prediction, we want negative values (consumption) to be high scores.
        So we return -log2FC to make consumption positive.

        Args:
            assertion_code: One of 'ND', 'D', '', or numeric string
            log2fc: Log2 fold change value (if available)

        Returns:
            Score where positive = consumption, negative = production, NaN = not detected
        """
        if pd.isna(assertion_code) or assertion_code == '':
            # Detected but not significant
            return 0.0

        assertion_code = str(assertion_code).strip()

        if assertion_code == 'ND':
            # Not detected - exclude from averaging
            return np.nan

        if assertion_code == 'D':
            # Detected but not significant
            return 0.0

        # Try to parse as number (the assertion code itself might be the significance level)
        try:
            # If assertion code is numeric, use log2FC
            int(assertion_code)
            if pd.notna(log2fc):
                # Negative log2FC = consumption (depletion) -> positive score
                return -float(log2fc)
            else:
                return 0.0
        except ValueError:
            # Unknown assertion code, treat as not significant
            return 0.0

    def load_data(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load and combine training data from multiple CSV files.

        Args:
            file_paths: List of paths to WoM format CSV files

        Returns:
            Combined DataFrame with columns: organism, compound, score
        """
        all_data = []

        for file_path in file_paths:
            print(f"Loading {Path(file_path).name}...")
            df = pd.read_csv(file_path)

            # Parse each row
            for _, row in df.iterrows():
                organism = row['Organism Id']
                compound = row['Compound Id']
                assertion_code = row['Assertion Code']
                log2fc = row.get('Log2 Fold Change', np.nan)

                # Parse to numerical score
                score = self.parse_assertion_code(assertion_code, log2fc)

                # Skip ND entries
                if pd.notna(score):
                    all_data.append({
                        'organism': organism,
                        'compound': compound,
                        'score': score
                    })

            print(f"  Loaded {len(all_data)} observations")

        return pd.DataFrame(all_data)

    def load_genus_mapping(self, mapping_file: str):
        """
        Load genus mapping from biggs_training_data.json.

        Args:
            mapping_file: Path to biggs_training_data.json
        """
        print(f"Loading genus mapping from {Path(mapping_file).name}...")
        with open(mapping_file, 'r') as f:
            data = json.load(f)

        # Extract unique organism -> genus mappings
        for entry in data:
            organism = entry['isolate']
            genus = entry.get('genus')
            if organism not in self.organism_to_genus and pd.notna(genus) and genus != "":
                self.organism_to_genus[organism] = genus

        print(f"  Loaded mappings for {len(self.organism_to_genus)} organisms")

    def fit(self, file_paths: List[str], genus_mapping_file: Optional[str] = None):
        """
        Train the baseline model by computing genus-level averages.

        Args:
            file_paths: List of paths to training data CSV files
            genus_mapping_file: Optional path to genus mapping JSON (e.g., biggs_training_data.json)
        """
        print("\n=== Training Genus-Average Baseline ===\n")

        # Load genus mapping if provided
        if genus_mapping_file:
            self.load_genus_mapping(genus_mapping_file)

        # Load all training data
        df = self.load_data(file_paths)

        print(f"\nTotal observations: {len(df)}")
        print(f"Unique organisms: {df['organism'].nunique()}")
        print(f"Unique compounds: {df['compound'].nunique()}")

        # Extract genus for each organism
        df['genus'] = df['organism'].apply(self.extract_genus)

        print(f"\nGenus extraction:")
        print(f"  Unique genera: {df['genus'].nunique()}")
        print(f"  Unknown organisms: {(df['genus'] == 'Unknown').sum()}")

        # Show genus distribution
        genus_counts = df['genus'].value_counts()
        print(f"\nTop genera by observation count:")
        for genus, count in genus_counts.head(10).items():
            organism_count = df[df['genus'] == genus]['organism'].nunique()
            print(f"  {genus}: {count} observations from {organism_count} organisms")

        # Get list of all metabolites
        self.metabolites = sorted(df['compound'].unique())
        print(f"\nMetabolites: {len(self.metabolites)}")

        # Compute genus-level averages
        print("\nComputing genus profiles...")

        for genus in df['genus'].unique():
            genus_df = df[df['genus'] == genus]

            # Compute average score for each metabolite
            genus_profile = {}
            for metabolite in self.metabolites:
                metabolite_scores = genus_df[genus_df['compound'] == metabolite]['score']
                if len(metabolite_scores) > 0:
                    # Use nanmean to exclude NaN values
                    avg_score = metabolite_scores.mean()
                    genus_profile[metabolite] = avg_score
                else:
                    # No data for this metabolite in this genus
                    genus_profile[metabolite] = 0.0

            self.genus_profiles[genus] = genus_profile

        print(f"  Computed profiles for {len(self.genus_profiles)} genera")

        # Compute population mean (fallback for unknown genera)
        print("\nComputing population mean...")
        population_profile = {}
        for metabolite in self.metabolites:
            metabolite_scores = df[df['compound'] == metabolite]['score']
            if len(metabolite_scores) > 0:
                avg_score = metabolite_scores.mean()
                population_profile[metabolite] = avg_score
            else:
                population_profile[metabolite] = 0.0

        self.population_mean = population_profile

        self.trained = True
        print("\n=== Training Complete ===\n")

    def predict(self, organism_name: str) -> Dict[str, float]:
        """
        Predict metabolite preference profile for an organism.

        Args:
            organism_name: Name of organism to predict for

        Returns:
            Dictionary mapping metabolite -> score
            Positive scores indicate consumption, negative indicate production
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")

        genus = self.extract_genus(organism_name)

        if genus in self.genus_profiles:
            return self.genus_profiles[genus].copy()
        else:
            # Fallback to population mean
            return self.population_mean.copy()

    def get_top_consumed(self, organism_name: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most consumed metabolites for an organism.

        Args:
            organism_name: Name of organism
            n: Number of top metabolites to return

        Returns:
            List of (metabolite, score) tuples, sorted by score descending
            Higher scores indicate stronger consumption
        """
        profile = self.predict(organism_name)

        # Sort by score descending (highest consumption first)
        sorted_metabolites = sorted(
            profile.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_metabolites[:n]

    def get_top_produced(self, organism_name: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most produced metabolites for an organism.

        Args:
            organism_name: Name of organism
            n: Number of top metabolites to return

        Returns:
            List of (metabolite, score) tuples, sorted by score ascending
            Lower (more negative) scores indicate stronger production
        """
        profile = self.predict(organism_name)

        # Sort by score ascending (lowest/most negative = highest production first)
        sorted_metabolites = sorted(
            profile.items(),
            key=lambda x: x[1],
            reverse=False
        )

        return sorted_metabolites[:n]

    def save_profiles(self, output_path: str):
        """
        Save genus profiles to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")

        output_data = {
            'genus_profiles': self.genus_profiles,
            'population_mean': self.population_mean,
            'metabolites': self.metabolites,
            'organism_to_genus': self.organism_to_genus
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved genus profiles to {output_path}")

    def load_profiles(self, input_path: str):
        """
        Load genus profiles from JSON file.

        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        self.genus_profiles = data['genus_profiles']
        self.population_mean = data['population_mean']
        self.metabolites = data['metabolites']
        self.organism_to_genus = data.get('organism_to_genus', {})
        self.trained = True

        print(f"\nLoaded genus profiles from {input_path}")
        print(f"  Genera: {len(self.genus_profiles)}")
        print(f"  Metabolites: {len(self.metabolites)}")
        print(f"  Organism mappings: {len(self.organism_to_genus)}")


def main():
    """
    Train the genus-average baseline model and save profiles.
    """
    # Define paths to training data
    data_dir = Path(__file__).parent.parent / "04_exometabolomics_data" / "processed"

    training_files = [
        str(data_dir / "biggs_nldm_wom_format.csv"),
        str(data_dir / "17member_nldm_wom_format.csv")
    ]

    genus_mapping_file = str(data_dir / "biggs_training_data.json")

    # Initialize and train model
    model = GenusAverageBaseline()
    model.fit(training_files, genus_mapping_file=genus_mapping_file)

    # Save profiles
    output_path = Path(__file__).parent / "genus_profiles.json"
    model.save_profiles(str(output_path))

    # Demonstrate predictions
    print("\n=== Example Predictions ===\n")

    test_organisms = [
        "BacillusOAE603",
        "ArthrobacterOAP107",
        "VariovoraxOAS795",
        "FW507-14D01",  # Unknown genus
    ]

    for organism in test_organisms:
        genus = model.extract_genus(organism)
        print(f"\nOrganism: {organism}")
        print(f"  Genus: {genus}")
        print(f"  Using profile: {'genus average' if genus in model.genus_profiles else 'population mean'}")

        # Get top consumed metabolites
        top_consumed = model.get_top_consumed(organism, n=5)
        print(f"\n  Top 5 consumed metabolites:")
        for metabolite, score in top_consumed:
            print(f"    {metabolite}: {score:.3f}")

        # Get top produced metabolites
        top_produced = model.get_top_produced(organism, n=5)
        print(f"\n  Top 5 produced metabolites:")
        for metabolite, score in top_produced:
            print(f"    {metabolite}: {score:.3f}")


if __name__ == "__main__":
    main()
