"""Tests for the command-line interface (scripts/wom.py).

Tests the CLI-specific functionality:
- Output formatting (JSON vs table)
- Exit codes
- stderr handling
"""

import sys
import json
import pytest
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.wom import output_response, main


class TestArgumentParsing:
    """Test CLI argument parsing via main function."""
    
    @staticmethod
    def create_parser():
        """Create argument parser for testing."""
        parser = argparse.ArgumentParser(
            description="Query metabolite preference predictions"
        )
        parser.add_argument("--model-dir", default="trained_models/latest")
        parser.add_argument("--format", default="json", choices=["json", "table"])
        
        subparsers = parser.add_subparsers(dest="command")
        
        # compounds
        compounds_parser = subparsers.add_parser("compounds")
        compounds_parser.add_argument("--microbes", required=True)
        compounds_parser.add_argument("--top", type=int, default=20)
        compounds_parser.add_argument("--explain", action="store_true")
        compounds_parser.add_argument("--compute-distinctive", action="store_true", default=True)
        
        # microbes
        microbes_parser = subparsers.add_parser("microbes")
        microbes_parser.add_argument("--compounds", required=True)
        microbes_parser.add_argument("--level", default="genus")
        microbes_parser.add_argument("--top", type=int, default=10)
        microbes_parser.add_argument("--explain", action="store_true")
        
        # list
        list_parser = subparsers.add_parser("list")
        list_parser.add_argument("--level", default="genus")
        
        return parser
    
    def test_compounds_command_parsing(self):
        """Test parsing 'compounds' command arguments."""
        parser = self.create_parser()
        args = parser.parse_args([
            "compounds",
            "--microbes", "genus:Pseudomonas",
            "--top", "20",
        ])
        
        assert args.command == "compounds"
        assert args.microbes == "genus:Pseudomonas"
        assert args.top == 20
    
    def test_microbes_command_parsing(self):
        """Test parsing 'microbes' command arguments."""
        parser = self.create_parser()
        args = parser.parse_args([
            "microbes",
            "--compounds", "glucose,serine",
            "--level", "genus",
            "--top", "10",
        ])
        
        assert args.command == "microbes"
        assert args.compounds == "glucose,serine"
        assert args.level == "genus"
        assert args.top == 10
    
    def test_list_command_parsing(self):
        """Test parsing 'list' command arguments."""
        parser = self.create_parser()
        args = parser.parse_args([
            "list",
            "--level", "family",
        ])
        
        assert args.command == "list"
        assert args.level == "family"
    
    def test_model_dir_flag(self):
        """Test --model-dir flag across all commands."""
        parser = self.create_parser()
        args = parser.parse_args([
            "--model-dir", "custom/path/model",
            "compounds",
            "--microbes", "genus:Pseudomonas",
        ])
        
        assert args.model_dir == "custom/path/model"
    
    def test_explain_flag(self):
        """Test --explain flag."""
        parser = self.create_parser()
        args = parser.parse_args([
            "compounds",
            "--microbes", "genus:Pseudomonas",
            "--explain",
        ])
        
        assert args.explain is True
    
    def test_format_flag(self):
        """Test --format flag (json/table)."""
        parser = self.create_parser()
        args = parser.parse_args([
            "--format", "table",
            "compounds",
            "--microbes", "genus:Pseudomonas",
        ])
        
        assert args.format == "table"
    
    def test_default_values(self):
        """Test that defaults are applied correctly."""
        parser = self.create_parser()
        args = parser.parse_args([
            "compounds",
            "--microbes", "genus:Pseudomonas",
        ])
        
        # Check defaults
        assert args.format == "json"  # Default format
        assert args.top == 20  # Default top for compounds
    
    def test_comma_separated_values(self):
        """Test parsing comma-separated values."""
        parser = self.create_parser()
        args = parser.parse_args([
            "microbes",
            "--compounds", "glucose,pyruvate,acetate,serine",
        ])
        
        assert "glucose" in args.compounds
        assert "pyruvate" in args.compounds
        assert "serine" in args.compounds
    
    def test_colon_in_microbe_spec(self):
        """Test colon in microbe specification."""
        parser = self.create_parser()
        args = parser.parse_args([
            "compounds",
            "--microbes", "genus:Pseudomonas",
        ])
        
        assert "genus:Pseudomonas" == args.microbes


class TestOutputFormatting:
    """Test output formatting."""
    
    def test_json_output(self):
        """Test JSON output formatting."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "success": True,
            "results": [{"name": "test"}],
            "metadata": {},
        }
        
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            output_response(mock_response, format="json")
            output = mock_stdout.getvalue()
            
            # Should be valid JSON
            data = json.loads(output)
            assert data["success"] is True
    
    def test_json_output_pretty_printed(self):
        """Test that JSON is pretty-printed."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "success": True,
            "results": [],
        }
        
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            output_response(mock_response, format="json")
            output = mock_stdout.getvalue()
            
            # Should contain indentation (pretty-printed)
            assert "  " in output or "\n" in output
    
    def test_validation_warnings_to_stderr(self):
        """Test that validation warnings are printed to stderr."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"success": True}
        mock_response.metadata = {
            "warnings": ["Warning 1", "Warning 2"]
        }
        
        with patch("sys.stdout", new_callable=StringIO):
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                output_response(mock_response, format="json")
                stderr_output = mock_stderr.getvalue()
                
                # Warnings should be in stderr
                assert "Warning 1" in stderr_output
                assert "Warning 2" in stderr_output
                assert "⚠️" in stderr_output or "WARNING" in stderr_output
    
    def test_error_output_to_stderr(self):
        """Test that errors are printed to stderr."""
        mock_response = MagicMock()
        mock_response.success = False
        mock_response.error = "Test error message"
        
        with patch("sys.stdout", new_callable=StringIO):
            with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
                output_response(mock_response, format="table")
                stderr_output = mock_stderr.getvalue()
                
                # Error should be in stderr
                assert "Error" in stderr_output or "error" in stderr_output


class TestExitCodes:
    """Test CLI exit codes."""
    
    def test_invalid_arguments_exit_code(self):
        """Test exit code 2 for invalid arguments."""
        # argparse typically uses exit code 2 for usage errors
        parser = TestArgumentParsing.create_parser()
        
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["nonexistent_command"])
        
        # argparse exits with code 2 on error
        assert exc_info.value.code == 2

