import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid


# UI profile should be model x [user input]

class StyleFeatureSpace:
    """A multidimensional feature space for text style profiles with dynamic vectorization."""

    DEFAULT_FEATURES: dict[str, list[str]] = {
        "Sentence Structure": [
            "Simple & Direct",
            "Balanced & Varied",
            "Complex & Layered",
        ],
        "Vocab & Dictation": [
            "Plain & Accessible",
            "Precise & Technical",
            "Elevated & Literary",
        ],
        "Tone": [
            "Warm & Personal",
            "Neutral & Objective",
            "Assertive & Opinionated",
        ],
        "Formality": ["Casual", "Conversational-Professional", "Formal"],
        "Rhythm & Pacing": [
            "Terse & Punchy",
            "Measured & Flowing",
            "Expansive & Meandering",
        ],
    }

    def __init__(self, features: Optional[dict[str, list[str]]] = None) -> None:
        self.features = features if features is not None else self.DEFAULT_FEATURES

        # Dynamically build bidirectional mappings for fast vectorization/devectorization
        self._str_to_int: dict[str, dict[str, int]] = {
            dim: {val: idx for idx, val in enumerate(options)} for dim, options in self.features.items()
        }
        self._int_to_str: dict[str, dict[int, str]] = {
            dim: {idx: val for idx, val in enumerate(options)} for dim, options in self.features.items()
        }

    # --- 1. Vectorize the Schema Definition ---
    def get_vectorized_schema(self) -> dict[str, list[int]]:
        """Replaces each string list in the feature dictionary with integer lists of 0 to n-1."""
        return {
            dim: list(range(len(options)))
            for dim, options in self.features.items()
        }

    # --- 2. Vectorize / Devectorize Individual Profiles ---
    def vectorize_profile(self, profile: dict[str, str]) -> list[int]:
        """Converts a string-based style profile dictionary into a 1D integer vector."""
        try:
            return [self._str_to_int[dim][val] for dim, val in profile.items()]

        except KeyError as e:
            raise ValueError(f"Invalid feature dimension or category value: {e}") from e

    def devectorize_profile(self, vector: list[int]) -> dict[str, str]:
        """Converts a 1D integer vector back into a human-readable style profile dictionary."""
        if len(vector) != len(self.features):
            raise ValueError(f"Expected vector of length {len(self.features)}, got {len(vector)}")
        return {
            dim: self._int_to_str[dim][idx] for dim, idx in zip(self.features.keys(), vector)
        }

    # --- 3. Generate 2D Feature Matrix ---
    def generate_feature_matrix(self, as_numpy: bool = False) -> list[list[int]] | Any:
        """Generates all combinatorial profiles as a 2D integer array (matrix of shape [total_combinations, num_dimensions])."""
        vectorized_options = self.get_vectorized_schema().values()
        matrix = [list(comb) for comb in itertools.product(*vectorized_options)]

        if as_numpy:
            try:
                import numpy as np
                return np.array(matrix)

            except ImportError:
                print("NumPy not installed. Returning standard Python list of lists.")

        return matrix

      
    

if __name__ == "__main__":
    pass