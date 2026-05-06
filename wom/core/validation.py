"""Input validation and matching for queries.

Provides consistent feedback about what inputs matched, what didn't,
and why - critical for user experience when dealing with typos and
missing data.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InputMatchResult:
    """Result of attempting to match a single user input."""
    requested: str  # What the user typed
    matched: Optional[str]  # What we actually used (None if no match)
    match_type: str  # 'exact', 'fuzzy', 'not_found', 'not_in_model'
    match_score: float  # 0.0 to 1.0 for fuzzy matches
    reason: Optional[str] = None  # Why it didn't match (if applicable)
    
    @property
    def success(self) -> bool:
        """Did this input successfully match something usable?"""
        return self.matched is not None and self.match_type != 'not_in_model'
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "requested": self.requested,
            "matched": self.matched,
            "match_type": self.match_type,
            "match_score": self.match_score,
            "success": self.success,
            "reason": self.reason,
        }


@dataclass
class InputValidationSummary:
    """Summary of input validation across all items."""
    total_requested: int
    total_matched: int
    total_failed: int
    match_rate: float  # total_matched / total_requested
    
    # Lists of results
    matched_items: list[InputMatchResult]
    failed_items: list[InputMatchResult]
    
    # Warnings/errors
    warnings: list[str]
    should_warn: bool  # True if match_rate < 0.8
    should_fail: bool  # True if match_rate == 0.0
    
    @classmethod
    def from_results(cls, results: list[InputMatchResult]) -> "InputValidationSummary":
        """Create summary from list of match results."""
        matched = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_requested = len(results)
        total_matched = len(matched)
        total_failed = len(failed)
        match_rate = total_matched / total_requested if total_requested > 0 else 0.0
        
        # Generate warnings
        warnings = []
        if total_failed > 0:
            warnings.append(
                f"{total_failed}/{total_requested} inputs did not match: "
                + ", ".join([f'"{r.requested}"' for r in failed[:5]])
                + (" ..." if len(failed) > 5 else "")
            )
        
        fuzzy_matches = [r for r in matched if r.match_type == 'fuzzy']
        if fuzzy_matches:
            warnings.append(
                f"{len(fuzzy_matches)} fuzzy matches (check for typos): "
                + ", ".join([f'"{r.requested}" → "{r.matched}"' for r in fuzzy_matches[:3]])
                + (" ..." if len(fuzzy_matches) > 3 else "")
            )
        
        return cls(
            total_requested=total_requested,
            total_matched=total_matched,
            total_failed=total_failed,
            match_rate=match_rate,
            matched_items=matched,
            failed_items=failed,
            warnings=warnings,
            should_warn=match_rate < 0.8,
            should_fail=match_rate == 0.0,
        )
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "total_requested": self.total_requested,
            "total_matched": self.total_matched,
            "total_failed": self.total_failed,
            "match_rate": round(self.match_rate, 3),
            "matched_items": [r.to_dict() for r in self.matched_items],
            "failed_items": [r.to_dict() for r in self.failed_items],
            "warnings": self.warnings,
        }
