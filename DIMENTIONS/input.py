import itertools
from typing import Any, Optional



class StyleFeatureSpace:
    """A multidimensional feature space for text style profiles with dynamic vectorization."""

    DEFAULT_FEATURES: dict[str, list[str]] = {
    "Structure and Information Density": [
        "Simple, direct sentence structures. Short paragraphs (1-2 sentences), low information density, frequent breaks and whitespace. One idea at a time with minimal layering.",
        "Balanced sentence variety and paragraph development. Standard 3-5 sentence paragraphs built around a single idea. Moderate information density with clear organization.",
        "Complex, layered sentence structures and dense paragraphs containing multiple related ideas. High information density, extended reasoning, and substantial conceptual packing before breaks.",
    ],

    "Voice, Formality and Delivery": [
        "Neutral, casual, terse, punchy, and economical. Minimal elaboration, frequent short bursts, conversational phrasing, and rapid point-to-point movement.",
        "Professional and conversational. Measured pacing, moderate explanation, smooth transitions, and a balance between clarity and personality.",
        "Formal or strongly expressive. Expansive pacing, extensive elaboration, visible conviction or evaluation, and willingness to explore ideas from multiple angles before concluding.",
    ],

    "Hedging and Directness": [
        "Highly direct and assertive. States claims plainly, uses few qualifiers, and avoids hedging language unless strictly necessary.",
        "Calibrated and balanced. Uses hedging only where uncertainty genuinely exists, while still making the core point clearly.",
        "Cautious and highly qualified. Frequently acknowledges uncertainty, alternative interpretations, limitations, and multiple angles before committing to a position.",
    ],

    "Vocabulary and Diction": [
        "Plain and accessible. Common words, everyday register, minimal jargon.",
        "Precise and technical when useful. Domain-appropriate terminology explained or contextualized.",
        "Elevated, literary, or highly specialized vocabulary. Emphasis on precision, nuance, and evocative word choice.",
    ],

    "Rhetorical Devices": [
        "Literal language. Few or no metaphors, analogies, or figurative devices.",
        "Occasional metaphor or analogy used selectively for clarity.",
        "Frequent and sustained figurative language, recurring analogies, and rhetorical framing woven through the text.",
    ],

    "Person and Perspective": [
        "Impersonal. Third-person perspective, avoids direct reader address and personal opinion.",
        "Mixed perspective. Uses first person for opinions and second person for guidance when helpful.",
        "Strong personal voice. First-person reasoning and direct conversational engagement with the reader throughout.",
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

