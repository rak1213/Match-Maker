import numpy as np
import match_making_visualizer as mmv
import config as CONFIG
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

def compute_similarity(embeddings_dict, reference_embedding):
    """Compute cosine similarity between a reference embedding and embeddings in a dictionary."""
    similarities = {}
    for name, emb in embeddings_dict.items():
        similarity = 1 - cosine(reference_embedding, emb) 
        similarities[name] = similarity
    return similarities

def rank_correlation(model1_embeddings, model2_embeddings, person_ref_embedding):
    """Compute Spearman's rank correlation between two sets of embeddings."""
    similarities_model1 = compute_similarity(model1_embeddings, model1_embeddings[person_ref_embedding])
    similarities_model2 = compute_similarity(model2_embeddings, model2_embeddings[person_ref_embedding])

    # Sorting the names based on similarity
    sorted_names_model1 = sorted(similarities_model1, key=similarities_model1.get, reverse=True)
    sorted_names_model2 = sorted(similarities_model2, key=similarities_model2.get, reverse=True)

    # Creating rank lists
    ranks_model1 = [sorted_names_model1.index(name) for name in similarities_model1]
    ranks_model2 = [sorted_names_model2.index(name) for name in similarities_model1]

    # Compute Spearman's rank correlation
    correlation, x = spearmanr(ranks_model1, ranks_model2)
    return correlation



if __name__ == "__main__":
    classmates_map = mmv.match_making_model.read_data('Dataset.csv')
    person_embeddings_minilm_l6_v2 = mmv.match_making_model.generate_embeddings(classmates_map,CONFIG.TRANSFORMER_MINILM_L6_V2)
    person_embeddings_mpnet_base_v2 = mmv.match_making_model.generate_embeddings(classmates_map,CONFIG.TRANSFORMER_MPNET_BASE_V2)
    correlation = rank_correlation(person_embeddings_minilm_l6_v2, person_embeddings_mpnet_base_v2, 'Rakshit Gupta')
    print(f"Spearman's rank correlation: {round(correlation * 100,3)}%")