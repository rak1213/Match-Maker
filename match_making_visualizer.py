import csv
import umap
import matplotlib.pyplot as plt
import json
import os
import config as CONFIG
import numpy as np
from collections import defaultdict
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


class match_making_model:
    def read_data(file_name):
        classmates_map = {}
        with open(file_name, newline='') as csv_file:
            classmates = csv.reader(csv_file, delimiter=',', quotechar='"')
            next(classmates)  # Skip the header row
            for row in classmates:
                name, paragraph = row
                classmates_map[name] = paragraph
        return classmates_map

    def generate_embeddings(class_val_map,transformer):
        # Generate sentence embeddings
        person_embeddings = {}
        sentence_trasformer_model = SentenceTransformer(transformer)
        paragraphs = list(class_val_map.values())
        embeddings = sentence_trasformer_model.encode(paragraphs)    
        # Create a dictionary to store embeddings for each person
        person_embeddings = {list(class_val_map.keys())[list(class_val_map.values()).index(paragraph)]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
        return person_embeddings


    # Reducing dimensionality of embedding data, scaling to coordinate domain/range
    def dimension_reduction(embedding_vector):
        reduced_data = []
        reducer = umap.UMAP(random_state=42)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(list(embedding_vector.values()))
        reduced_data = reducer.fit_transform(scaled_data)
        return reduced_data
    # Creating lists of coordinates with accompanying labels

    def save_plot_matches(reduced_data,file_name,person_embeddings):
        path = os.path.join(CONFIG.BASE_RESULTS, file_name)
        x = [row[0] for row in reduced_data]
        y = [row[1] for row in reduced_data]
        label = list(person_embeddings.keys())
        # Plotting and annotating data points
        plt.scatter(x,y)
        for i, name in enumerate(label):
            plt.annotate(name, (x[i], y[i]), fontsize="3")        
        plt.axis('off')
        plt.savefig(path, dpi=800)


    def all_top_match_people(classmates_map,person_embeddings):
        top_matches = {}
        all_personal_pairs = defaultdict(list)
        for person in classmates_map.keys():
            for person1 in classmates_map.keys():
                all_personal_pairs[person].append([spatial.distance.cosine(person_embeddings[person1], person_embeddings[person]), person1])

        for person in classmates_map.keys():
            top_matches[person] = sorted(all_personal_pairs[person], key=lambda x: x[1])
        return top_matches

    def save_embeddings_json(embeddings_dict,file_name):
        path = os.path.join(CONFIG.BASE_RESULTS, file_name+'.json')
        with open(path, 'w') as json_file:
            json.dump(embeddings_dict, json_file,cls=NumpyArrayEncoder)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

if __name__ == '__main__':
    transformer_minilm = CONFIG.TRANSFORMER_MINILM_L6_V2
    transformer_mpnet = CONFIG.TRANSFORMER_MPNET_BASE_V2
    transformer = transformer_minilm
    if(transformer == transformer_minilm ):
        img_name = 'visualization_minilm'
        embeddings_file_name = 'person_embeddings_minilm'
        red_embeddings_file_name = 'reduced_person_embeddings_minilm'
    elif(transformer == transformer_mpnet ):
        img_name = 'visualization_mpnet'
        embeddings_file_name = 'person_embeddings_mpnet'
        red_embeddings_file_name = 'reduced_person_embeddings_mpnet'

    classmates_map = match_making_model.read_data('Dataset.csv')
    person_embeddings = match_making_model.generate_embeddings(classmates_map,transformer)
    match_making_model.save_embeddings_json(person_embeddings,embeddings_file_name)
    reduced_embeddings_data = match_making_model.dimension_reduction(person_embeddings)
    match_making_model.save_embeddings_json(reduced_embeddings_data,red_embeddings_file_name)
    match_making_model.save_plot_matches(reduced_embeddings_data, img_name,person_embeddings)
    top_matches = match_making_model.all_top_match_people(classmates_map,person_embeddings)
