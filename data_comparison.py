import numpy as np
import match_making_visualizer as mmv
import match_making_visualizer as mmv
import config as CONFIG
import json
import os
from scipy.spatial.distance import cosine


def load_embeddings(file_path):
    """
    Load embeddings from a JSON file.

    Args:
        file_path (string): Path of the JSON file.

    Returns:
        dict: A dictionary mapping classmate names (keys) to their sentence embeddings (values).
    """
    with open(file_path, "r") as file:
        embeddings = json.load(file)
    return embeddings


def compare_embeddings(old_embeddings, new_embeddings):
    """
    Compare embeddings using cosine similarity

    Args:
        old_embeddings (dict): Embeddings from actual dataset for selected people
        new_embeddings (dict): Embeddings from modified dataset after rephrasing or adding synonms or antonyms for selected people

    Returns:
        dict: A dictionary mapping classmate names (keys) to their different sentence embeddings similarities (values).
    """
    similarities = []
    for name in old_embeddings:
        old_emb = np.array(old_embeddings[name])
        new_emb = np.array(new_embeddings[name])
        similarity = 1 - cosine(old_emb, new_emb)
        similarity = round(similarity, 3)
        similarities.append((name, similarity))
    return similarities


if __name__ == "__main__":
    """
    Script for comparing the embeddings formed by rephrasing or replacing a word with synonyms or antonyms.

    This block selects 4 people from the dataset and modifies their sentences by include replacing a key word 
    with a synonym, antonym and changing the phrasing of the sentence. Comparing their embeddings with the actual
    embeddings and finding the cosine similarities to know how the embeddings got affected by incorporating such changes
    and getting the results.

    """
    selected_list = [
        "Rakshit Gupta",
        "Neeyati Mehta",
        "Sylvester Terdoo",
        "Tejasvi Bhutiyal",
    ]
    classmates_map_data_change = {}
    classmates_map_data_change[
        "Rakshit Gupta"
    ] = "I have a passion for journeying to new destinations, uncovering unfamiliar places, and relish engaging in pastimes such as swimming and basketball."
    classmates_map_data_change["Neeyati Mehta"] = "I like napping."
    classmates_map_data_change["Sylvester Terdoo"] = "i enjoy spending time inside"
    classmates_map_data_change[
        "Tejasvi Bhutiyal"
    ] = "I love to binge watch series and explore new places."
    person_modified_embeddings = mmv.MatchMakingModel.generate_embeddings(
        classmates_map_data_change, CONFIG.TRANSFORMER_MINILM_L6_V2
    )

    mmv.MatchMakingModel.save_embeddings_json(
        person_modified_embeddings, "modified_embeddings_minilm"
    )
    old_embeddings_path = os.path.join(
        CONFIG.BASE_RESULTS, "person_embeddings_minilm.json"
    )

    person_embeddings = load_embeddings(old_embeddings_path)
    person_selected_embeddings = {
        key: val for key, val in person_embeddings.items() if key in selected_list
    }

    assert (
        person_selected_embeddings.keys() == person_modified_embeddings.keys()
    ), "Embedding dictionaries must have the same keys."

    similarities = compare_embeddings(
        person_selected_embeddings, person_modified_embeddings
    )

    for name, similarity in similarities:
        print(
            f"Similarity between sentence of {name} and Modifed {name}: {similarity * 100}%"
        )
