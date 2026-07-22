import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit  # numerically stable sigmoid

from DIMENTIONS.input import StyleFeatureSpace
from ALGO.pref_learn_algo import MythosNonLinearAlgo
from LLM_API.llm_api import get_llm

# UI profile should be model x [user input]
# 

NEW_USER = True
round_num = 0

def main():

    style_space = StyleFeatureSpace()

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


    while True:
        round_num += 1
        a_vect, b_vect = algo.get_comparison()

        a_devect = style_space.devectorize_profile(a_vect)
        b_devect = style_space.devectorize_profile(b_vect)

        get_llm()
    
    
    

    
      
    

if __name__ == "__main__":
    pass