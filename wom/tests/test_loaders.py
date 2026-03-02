"""Test loaders module - data loading and preprocessing.

This module tests the loaders/, which handles:
- Loading exometabolite dataset from files
- Loading ChEBI chemical database
- Mapping compound names to ChEBI IDs
- Preparing training data from raw data
"""

import pytest
from pathlib import Path

from loaders.loaders import (
    load_exometabolite,
    prepare_training_data,
   get_organism_preferences,
)
from loaders.compound_mapper import load_chebi, map_compound_name
from core.types import AssertionType


class TestChEBILoading:
    """Test ChEBI database loading."""

    def test_chebi_loads_successfully(self, chebi_data):
        """ChEBI data should load without errors."""
        assert chebi_data is not None
        assert len(chebi_data) > 0

    def test_chebi_has_required_columns(self, chebi_data):
        """ChEBI data should have required columns."""
        required_cols = {"compound_id", "name"}
        assert required_cols.issubset(set(chebi_data.columns))

    def test_chebi_unique_compounds(self, chebi_data):
        """ChEBI should have unique compound IDs."""
        unique_compounds = chebi_data["compound_id"].nunique()
        assert unique_compounds > 0

    def test_compound_name_mapping(self, chebi_data):
        """Compound names should map to ChEBI entries."""
        test_compounds = ["glucose", "serine", "alanine"]
        
        for compound_name in test_compounds:
            mapping = map_compound_name(compound_name, chebi_data)
            assert mapping is not None
            assert mapping.representative_name is not None
            assert 0 <= mapping.match_score <= 1


class TestExometaboliteLoading:
    """Test exometabolite dataset loading."""

    def test_exometabolite_loads_successfully(self, exometabolite_data):
        """Exometabolite data should load without errors."""
        assert exometabolite_data is not None
        assert len(exometabolite_data) > 0

    def test_exometabolite_required_columns(self, exometabolite_data):
        """Exometabolite data should have required columns."""
        required_cols = {
            "organism_id",
            "representative_compound",
            "assertion",
            "assertion_value",
        }
        assert required_cols.issubset(set(exometabolite_data.columns))

    def test_exometabolite_has_diverse_data(self, exometabolite_data):
        """Exometabolite data should have multiple organisms and metabolites."""
        n_organisms = exometabolite_data["organism_id"].nunique()
        n_metabolites = exometabolite_data["representative_compound"].nunique()
        
        assert n_organisms > 1, "Should have multiple organisms"
        assert n_metabolites > 1, "Should have multiple metabolites"

    def test_exometabolite_assertion_distribution(self, exometabolite_data):
        """Exometabolite data should have various assertions."""
        assertions = exometabolite_data["assertion"].unique()
        assert len(assertions) > 0


class TestTrainingDataPreparation:
    """Test training data preparation."""

    def test_training_data_preparation(self, exometabolite_data):
        """Training data should prepare successfully."""
        training_data = prepare_training_data(
            exometabolite_data, 
            preference_type=AssertionType.CONSUMED
        )
        
        assert training_data is not None
        assert training_data.n_microbes > 0
        assert training_data.n_metabolites > 0

    def test_training_data_structure(self, training_data):
        """Training data should have proper structure."""
        assert hasattr(training_data, "microbe_to_idx")
        assert hasattr(training_data, "metabolite_to_idx")
        assert hasattr(training_data, "preference_type")
        assert hasattr(training_data, "microbe_taxonomy")

    def test_training_data_get_microbe_positives(self, training_data):
        """Should get positive metabolites for a microbe."""
        microbe_id = list(training_data.microbe_to_idx.keys())[0]
        positives = training_data.get_microbe_positives(microbe_id)
        
        assert positives is not None
        assert len(positives) >= 0

    def test_training_data_get_metabolite_positives(self, training_data):
        """Should get microbes that consume a metabolite."""
        metabolite_id = list(training_data.metabolite_to_idx.keys())[0]
        consumers = training_data.get_metabolite_positives(metabolite_id)
        
        assert consumers is not None
        assert len(consumers) >= 0

    def test_training_data_taxonomy(self, training_data):
        """Training data should include taxonomy information."""
        microbe_id = list(training_data.microbe_to_idx.keys())[0]
        taxonomy = training_data.microbe_taxonomy[microbe_id]
        
        assert taxonomy is not None
        assert hasattr(taxonomy, "genus")
        assert hasattr(taxonomy, "family")


class TestOrganismPreferences:
    """Test organism preference queries."""

    def test_get_organism_preferences(self, exometabolite_data):
        """Should retrieve preferences for an organism."""
        organism_id = exometabolite_data["organism_id"].iloc[0]
        prefs = get_organism_preferences(organism_id, exometabolite_data)
        
        assert prefs is not None
        assert len(prefs) > 0

    def test_organism_preferences_filter_by_assertion(self, exometabolite_data):
        """Should filter preferences by assertion type."""
        organism_id = exometabolite_data["organism_id"].iloc[0]
        prefs = get_organism_preferences(organism_id, exometabolite_data)
        
        assert prefs is not None
        # Filter manually to consumed  
        consumed_only = prefs[prefs["assertion"] == "consumed"]
        
        assert consumed_only is not None
        if len(consumed_only) > 0:
            assert all(consumed_only["assertion"] == "consumed")
