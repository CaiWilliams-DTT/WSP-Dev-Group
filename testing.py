import os
import itertools
import textwrap
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid

from DIMENTIONS.input import StyleFeatureSpace
from ALGO.pref_learn_algo import MythosLinearAlgo, MythosNonLinearAlgo
from LLM_API.llm_api import get_llm, style_dict_to_guide


def stylefeaturespace_test():
    style_space = StyleFeatureSpace()
    # 1. Get the vectorized schema (dict values replaced with 0 to n-1)
    print("Vectorized Schema:")
    print(style_space.get_vectorized_schema())
    # Output: {'Sentence Structure': [0, 1, 2], 'Vocab & Dictation': [0, 1, 2], ...}

    # 2. Vectorize a specific text profile
    sample_profile = {
        "Sentence Structure": "Complex & Layered",  # Index 2
        "Vocab & Dictation": "Precise & Technical",  # Index 1
        "Tone": "Neutral & Objective",  # Index 1
        "Formality": "Formal",  # Index 2
        "Rhythm & Pacing": "Terse & Punchy",  # Index 0
    }
    vector = style_space.vectorize_profile(sample_profile)
    print("\nVectorized Profile:", vector)
    # Output: [2, 1, 1, 2, 0]

    # 3. Decode the vector back to string labels
    print("Devectorized Back:", style_space.devectorize_profile(vector))

    # 4. Generate the complete 2D integer matrix (ideal for ML inputs)
    feature_matrix = style_space.generate_feature_matrix(as_numpy=True)
    print(f"\nFeature Matrix Shape: {feature_matrix.shape[0]} x {feature_matrix.shape[1]}")
    print("First 3 rows of feature matrix:")
    for row in feature_matrix[:3]:
        print(row)

def algo_test_old():
    style_space = StyleFeatureSpace()
    feature_matrix = style_space.generate_feature_matrix(as_numpy=True)


    # 5. Interactive active-preference loop over the full style population
    learner = MythosLinearAlgo(feature_matrix)
    dims = list(style_space.features.keys())
    label_width = max(len(d) for d in dims)

    print("\n" + "=" * 70)
    print(" ACTIVE STYLE PREFERENCE LEARNING (linear utility model)")
    print(f" population: {len(learner.population)} profiles,"
          f" {learner._L} slope weights")
    print(" answer 'A' or 'B' to state a preference, 'Q' to quit.")
    print("=" * 70)

    round_num = 0
    while True:
        round_num += 1
        a_vect, b_vect = learner.get_comparison(learner.population, learner.scores)
        profile_a = style_space.devectorize_profile(a_vect)
        profile_b = style_space.devectorize_profile(b_vect)

        print(f"\n--- QUERY #{round_num} " + "-" * 50)
        for dim in dims:
            mark = "  <-- differs" if profile_a[dim] != profile_b[dim] else ""
            print(f"  {dim:<{label_width}} | A: {profile_a[dim]:<28}"
                  f" B: {profile_b[dim]:<28}{mark}")

        answer = None
        while answer not in ("A", "B", "Q"):
            try:
                answer = input(" your choice [A/B/Q] > ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                answer = "Q"
            if answer not in ("A", "B", "Q"):
                print("   please type 'A', 'B' or 'Q'.")
        if answer == "Q":
            break

        learner.update_score(a_vect, b_vect, answer)

        # Score tracking table: posterior slope (mean ± std) per dimension.
        # A positive slope means utility rises with the option index, so
        # under the linear model the implied favourite of each dimension is
        # always an endpoint option.
        weight_std = np.sqrt(np.clip(np.diag(learner._Sigma), 0.0, None))
        records = []
        for d_i, dim in enumerate(dims):
            opts = style_space.features[dim]
            slope = float(learner._mu[d_i])
            if abs(slope) < 0.005:          # rounds to 0.00 - treat as no signal
                favourite = "- (no clear signal yet)"
            else:
                favourite = opts[-1] if slope > 0 else opts[0]
            records.append({
                "dimension": dim,
                "slope": f"{slope:+.2f}",
                "±std": f"{weight_std[d_i]:.2f}",
                "implied favourite": favourite,
            })
        print("\n score tracking table (per-dimension slopes; utility is linear in the option index):")
        print(pd.DataFrame(records).to_string(index=False))

        # Live leaderboard: top-scoring profiles under the current posterior.
        top = np.argsort(learner.scores)[::-1][:3]
        print(f"\n current top profiles after {len(learner.past_comparisons)} comparisons:")
        for rank, idx in enumerate(top, start=1):
            decoded = style_space.devectorize_profile(learner.population[idx])
            summary = " | ".join(decoded[d] for d in dims)
            print(f"  #{rank}  score {learner.scores[idx]:+.2f}"
                  f" (±{learner.score_stds[idx]:.2f})  {summary}")

    print("\n session ended.")
    if learner.past_comparisons:
        best_idx = int(np.argmax(learner.scores))
        print(" best-guess style profile:")
        for dim, val in style_space.devectorize_profile(learner.population[best_idx]).items():
            print(f"   {dim:<{label_width}} : {val}")


NEW_USER = True
run = True


def algo_test():

    def print_comparison(a_response: str, b_response: str, round_num: int) -> None:
        """Print the two responses side-by-side (sequentially) for comparison."""
        
        CLI_WIDTH = 100

        print("=" * CLI_WIDTH)
        print(f"Round {round_num}")
        print("=" * CLI_WIDTH)

        print("\nOPTION A")
        print("-" * CLI_WIDTH)
        print(textwrap.fill(a_response, width=CLI_WIDTH))

        print("\nOPTION B")
        print("-" * CLI_WIDTH)
        print(textwrap.fill(b_response, width=CLI_WIDTH))

        print("=" * CLI_WIDTH)

    def get_user_preference() -> str:
        """Prompt until the user enters a valid choice."""

        while True:
            choice = input("\nWhich response do you prefer? [A/B/Q]: ").strip().upper()

            if choice in {"A", "B", "Q"}:
                return choice

            print("Invalid choice. Please enter A, B or Q.")
    
    style_space = StyleFeatureSpace()
    round_num = 0

    # Past user profile? (dropdown in UI -> create new profile setting)
    # Generate vectorised feature matrix
    if NEW_USER:
        feature_matrix = style_space.generate_feature_matrix(as_numpy=True)
        algo = MythosNonLinearAlgo(vectors=feature_matrix, past_scores=False)
        
    else:
        # Load past profiles
        # feature_matrix, user_scores = 
        # algo = MythosNonLinearAlgo(vectors=feature_matrix, past_scores=user_scores)
        pass

    while run:
        os.system("cls" if os.name == "nt" else "clear")

        round_num += 1

        a_vect, b_vect = algo.get_comparison()

        a_styleguide = style_dict_to_guide(style_space.devectorize_profile(a_vect))
        b_styleguide = style_dict_to_guide(style_space.devectorize_profile(b_vect))

        a_resp = get_llm(a_styleguide)
        b_resp = get_llm(b_styleguide)

        print_comparison(a_resp, b_resp, round_num)

        user_preference = get_user_preference()

        if user_preference == "Q":
            break

        algo.update_score(a_vect, b_vect, user_preference)

        


if __name__ == '__main__':
    # stylefeaturespace_test()
    algo_test()
    pass