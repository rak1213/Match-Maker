import umap
import numpy as np
import json
import optuna
import os
import config as CONFIG
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

def load_embeddings(file_path):
    """Load embeddings from a JSON file."""
    with open(file_path, 'r') as file:
        embeddings = json.load(file)
    return embeddings

def umap_objective(trial):
    # Define the hyperparameters to tune
    n_neighbors = trial.suggest_int('n_neighbors', 2, 50)
    min_dist = trial.suggest_float('min_dist', 0.0, 0.9)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=CONFIG.RANDOM_STATE)

    path = os.path.join(CONFIG.BASE_RESULTS,'person_embeddings_minilm.json')
    embeddings_dict = load_embeddings(path)
    embeddings = np.array(list(embeddings_dict.values()))
    embedding_2d = reducer.fit_transform(embeddings)

    # Compute Spearman rank correlations
    correlations = []
    for i in range(len(embeddings)):
        # Compute cosine similarities for high-dimensional embeddings
        cos_similarities = np.array([1 - cosine(embeddings[i], emb) for emb in embeddings])

        # Compute Euclidean distances in the 2D space
        euclidean_dists = np.linalg.norm(embedding_2d - embedding_2d[i], axis=1)
        metric = trial.suggest_categorical('metric', ['euclidean', 'cosine'])

        # Compute rankings
        cos_ranks = np.argsort(cos_similarities)
        euclidean_ranks = np.argsort(euclidean_dists)

        # Compute and store Spearman rank correlation
        correlation, _ = spearmanr(cos_ranks, euclidean_ranks)
        correlations.append(correlation)

    # Return the average correlation
    return np.mean(correlations)


if __name__ == '__main__':  
    study = optuna.create_study(direction='maximize')
    study.optimize(umap_objective, n_trials=100)
    best_params = study.best_params
    print("Best Parameters:", best_params)