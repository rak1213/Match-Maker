import numpy as np
import match_making_visualizer as mmv
from scipy.spatial.distance import cosine
import match_making_visualizer as mmv
import config as CONFIG
import json

def load_embeddings(file_path):
    """Load embeddings from a JSON file."""
    with open(file_path, 'r') as file:
        embeddings = json.load(file)
    return embeddings

def compare_embeddings(old_embeddings, new_embeddings):
    """Compare embeddings using cosine similarity."""
    similarities = []
    for name in old_embeddings:
        old_emb = np.array(old_embeddings[name])
        new_emb = np.array(new_embeddings[name])
        similarity = 1 - cosine(old_emb, new_emb)
        similarity = round(similarity,3)
        similarities.append((name, similarity))
    return similarities


if __name__ == '__main__':  
    selected_list = ['Rakshit Gupta','Neeyati Mehta','Sylvester Terdoo','Tejasvi Bhutiyal']
    classmates_map_data_change ={}
    classmates_map_data_change['Rakshit Gupta']= 'I have a passion for journeying to new destinations, uncovering unfamiliar places, and relish engaging in pastimes such as swimming and basketball.'
    classmates_map_data_change['Neeyati Mehta'] = 'I like napping.'
    classmates_map_data_change['Sylvester Terdoo'] ='i enjoy spending time inside'
    classmates_map_data_change['Tejasvi Bhutiyal'] ='I love to binge watch series and explore new places.'
    person_modified_embeddings = mmv.match_making_model.generate_embeddings(classmates_map_data_change)
    
    mmv.match_making_model.save_embeddings_json(person_modified_embeddings,CONFIG.MOD_PERSON_EMBEDDING_DATA)
    old_embeddings_path = CONFIG.PERSON_EMBEDDING_DATA 


    person_embeddings = load_embeddings(old_embeddings_path)
    person_selected_embeddings = { key:val for key,val in person_embeddings.items() if key in selected_list}
    # Ensure the same keys are present in both dictionaries
    assert person_selected_embeddings.keys() == person_modified_embeddings.keys(), "Embedding dictionaries must have the same keys."

    # Compare embeddings
    similarities = compare_embeddings(person_selected_embeddings, person_modified_embeddings)

    # Output the results
    for name, similarity in similarities:
        print(f'Similarity between sentence of {name} and Modifed {name}: {similarity * 100}%')
    