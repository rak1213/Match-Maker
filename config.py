"""
This script segment sets up the initial configurations and constants for a matchmaking model.

It includes:
- Suppressing warnings for cleaner output.
- Defining base paths and transformer model names for use in embedding generation.
- A flag for controlling whether embeddings should be saved.
- Setting a random state for reproducibility in processes like dimensionality reduction and model training.

Note:
    The random state can be changed to different values for experimenting with different outcomes in model behavior.
"""

import warnings

warnings.simplefilter("ignore")

BASE_RESULTS = "Results"
TRANSFORMER_MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"
TRANSFORMER_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"

SAVE_EMBEDDINGS = 0

#RANDOM_STATE = 42
RANDOM_STATE = 0
#RANDOM_STATE = 23
