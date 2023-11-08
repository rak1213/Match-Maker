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


class MatchMakingModel:
    """
    This class represents a matchmaking model that processes a dataset of classmates,
    generates sentence embeddings from their descriptions, and identifies potential matches based on cosine similarity.
    """
    def read_data(file_name):
        """
        Reads data from a CSV file and maps each classmate's name to their respective paragraph.

        Args:
            file_name (str): The name of the CSV file to read from.

        Returns:
            dict: A dictionary mapping classmate names (keys) to their paragraphs (values).
        """
        classmates_map = {}
        with open(file_name, newline="") as csv_file:
            classmates = csv.reader(csv_file, delimiter=",", quotechar='"')
            next(classmates)  
            for row in classmates:
                name, paragraph = row
                classmates_map[name] = paragraph
        return classmates_map

    def generate_embeddings(class_val_map, transformer):
        """
        Generates sentence embeddings for each paragraph associated with a classmate.

        Args:
            class_val_map (dict): A dictionary mapping classmate names to their paragraphs.
            transformer (str): The transformer model to be used for generating embeddings.

        Returns:
            dict: A dictionary mapping classmate names (keys) to their sentence embeddings (values).
        """
    
        person_embeddings = {}
        sentence_trasformer_model = SentenceTransformer(transformer)
        paragraphs = list(class_val_map.values())
        embeddings = sentence_trasformer_model.encode(paragraphs)
        person_embeddings = {
            list(class_val_map.keys())[
                list(class_val_map.values()).index(paragraph)
            ]: embedding
            for paragraph, embedding in zip(paragraphs, embeddings)
        }
        return person_embeddings


    def dimension_reduction(embedding_vector):
        """
        Reduces the dimensionality of embedding vectors using UMAP and scales them.

        Args:
            embedding_vector (dict): A dictionary of embedding vectors.

        Returns:
            list: A list of reduced and scaled embedding vectors.
        """
        reduced_data = []
        reducer = umap.UMAP(random_state=42)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(list(embedding_vector.values()))
        reduced_data = reducer.fit_transform(scaled_data)
        return reduced_data

    # Creating lists of coordinates with accompanying labels

    def save_plot_matches(reduced_data, file_name, person_embeddings):
        """
        Saves a scatter plot of the reduced embedding data with annotations.

        Args:
            reduced_data (list): The reduced and scaled embedding data.
            file_name (str): The name of the file to save the plot to.
            person_embeddings (dict): A dictionary of person embeddings.

        Note:
            The plot is saved to a predefined path based on configuration settings.
        """
        path = os.path.join(CONFIG.BASE_RESULTS, file_name)
        x = [row[0] for row in reduced_data]
        y = [row[1] for row in reduced_data]
        label = list(person_embeddings.keys())
        plt.scatter(x, y)
        for i, name in enumerate(label):
            plt.annotate(name, (x[i], y[i]), fontsize="3")
        plt.axis("off")
        plt.savefig(path, dpi=800)

    def all_top_match_people(classmates_map, person_embeddings):
        """
        Finds and ranks the top matching classmates for each individual based on embeddings.

        Args:
            classmates_map (dict): A dictionary mapping classmate names to their paragraphs.
            person_embeddings (dict): A dictionary of person embeddings.

        Returns:
            dict: A dictionary mapping each person to a sorted list of their top matches.
        """
        top_matches = {}
        all_personal_pairs = defaultdict(list)
        for person in classmates_map.keys():
            for person1 in classmates_map.keys():
                all_personal_pairs[person].append(
                    [
                        spatial.distance.cosine(
                            person_embeddings[person1], person_embeddings[person]
                        ),
                        person1,
                    ]
                )

        for person in classmates_map.keys():
            top_matches[person] = sorted(all_personal_pairs[person], key=lambda x: x[1])
        return top_matches

    def save_embeddings_json(embeddings_dict, file_name):
        """
        Saves the embeddings dictionary as a JSON file.

        Args:
            embeddings_dict (dict): The dictionary of embeddings to save.
            file_name (str): The base name of the file to save the embeddings to.

        Note:
            The file is saved to a predefined path based on configuration settings.
        """
        path = os.path.join(CONFIG.BASE_RESULTS, file_name + ".json")
        with open(path, "w") as json_file:
            json.dump(embeddings_dict, json_file, cls=NumpyArrayEncoder)


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        Custom JSON encoder for numpy arrays.

        Args:
            obj (any): The object to encode.

        Returns:
            list: A list representation of the numpy array if it is an instance of numpy.ndarray.
            Otherwise, it uses the default JSON encoder for the object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    """
    Main execution block for the matchmaking model.

    This block initializes transformer models, determines file names based on the chosen transformer,
    reads classmate data from a CSV file, generates embeddings for each classmate using the selected transformer,
    performs dimensionality reduction on the embeddings, saves the embeddings and reduced embeddings as JSON files,
    plots and saves the reduced embeddings, and finally calculates the top matches for each classmate.

    Note:
        The block supports two transformer models: 'all-MiniLM-L6-v2' and 'all-mpnet-base-v2'.
        It uses these models to generate embeddings, reduce their dimensions, and visualize them in a scatter plot.
        It also calculates and stores the top matches for each individual based on their embeddings.
    """
    transformer_minilm = CONFIG.TRANSFORMER_MINILM_L6_V2
    transformer_mpnet = CONFIG.TRANSFORMER_MPNET_BASE_V2
    transformer = transformer_minilm
    if transformer == transformer_minilm:
        img_name = "visualization_minilm"
        embeddings_file_name = "person_embeddings_minilm"
        red_embeddings_file_name = "reduced_person_embeddings_minilm"
    elif transformer == transformer_mpnet:
        img_name = "visualization_mpnet"
        embeddings_file_name = "person_embeddings_mpnet"
        red_embeddings_file_name = "reduced_person_embeddings_mpnet"

    classmates_map = MatchMakingModel.read_data("Dataset.csv")
    person_embeddings = MatchMakingModel.generate_embeddings(
        classmates_map, transformer
    )
    reduced_embeddings_data = MatchMakingModel.dimension_reduction(person_embeddings)
    if (CONFIG.SAVE_EMBEDDINGS == 1):
        MatchMakingModel.save_embeddings_json(person_embeddings, embeddings_file_name)
        MatchMakingModel.save_embeddings_json(
            reduced_embeddings_data, red_embeddings_file_name
        )
    MatchMakingModel.save_plot_matches(
        reduced_embeddings_data, img_name, person_embeddings
    )
    top_matches = MatchMakingModel.all_top_match_people(
        classmates_map, person_embeddings
    )

