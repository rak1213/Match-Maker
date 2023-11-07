import os

BASE_RESULTS = 'Results'
TRANSFORMER_MINILM_L6_V2 = 'sentence-transformers/all-MiniLM-L6-v2'
TRANSFORMER_MPNET_BASE_V2 = 'sentence-transformers/all-mpnet-base-v2'

VISUALIZATION_IMAGE_MINILM_L6_V2 = os.path.join(BASE_RESULTS,'visualization_minilm.png')
VISUALIZATION_IMAGE_MPNET_BASE_V2 = os.path.join(BASE_RESULTS,'visualization_mpnet.png')
PERSON_EMBEDDING_DATA = os.path.join(BASE_RESULTS,'person_embeddings.json')
DIM_RED_PERSON_EMBEDDING_DATA = os.path.join(BASE_RESULTS,'dim_red_person_embeddings.json')
MOD_PERSON_EMBEDDING_DATA = os.path.join(BASE_RESULTS,'modified_person_embeddings.json')
