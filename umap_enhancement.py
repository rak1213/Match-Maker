import umap
import numpy as np
import json
import optuna
import os
import config as CONFIG
import match_making_visualizer as mmv
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


def load_embeddings(file_path):
    """
    Loads embeddings from a JSON file.

    Args:
        file_path (str): The file path of the JSON file containing the embeddings.

    Returns:
        dict: A dictionary containing the loaded embeddings.
    """
    with open(file_path, "r") as file:
        embeddings = json.load(file)
    return embeddings


def umap_objective(trial):
    """
    Objective function for optimizing UMAP hyperparameters using Optuna.

    Args:
        trial (optuna.trial.Trial): A trial instance from Optuna.

    Returns:
        float: The mean Spearman correlation coefficient across all pairs of embeddings.

    Note:
        This function suggests hyperparameters (n_neighbors, min_dist, metric) for UMAP,
        loads embeddings, applies dimensionality reduction, and calculates the mean Spearman
        correlation coefficient between cosine similarity ranks and Euclidean distance ranks of the embeddings.
    """
    n_neighbors = trial.suggest_int("n_neighbors", 2, 50)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.9)
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine"])
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=CONFIG.RANDOM_STATE,
    )

    path = os.path.join(CONFIG.BASE_RESULTS, "person_embeddings_minilm.json")
    embeddings_dict = load_embeddings(path)
    embeddings = np.array(list(embeddings_dict.values()))
    embedding_2d = reducer.fit_transform(embeddings)

    correlations = []
    for i in range(len(embeddings)):
        cos_similarities = np.array(
            [1 - cosine(embeddings[i], emb) for emb in embeddings]
        )

        euclidean_dists = np.linalg.norm(embedding_2d - embedding_2d[i], axis=1)

        cos_ranks = np.argsort(cos_similarities)
        euclidean_ranks = np.argsort(euclidean_dists)

        correlation, _ = spearmanr(cos_ranks, euclidean_ranks)
        correlations.append(correlation)

    return np.mean(correlations)


if __name__ == "__main__":
    """
    Main execution block to optimize UMAP hyperparameters and save a plot of the optimized embeddings.

    Note:
        This block creates a study object for hyperparameter optimization, optimizes it,
        prints the best parameters, loads embeddings, applies the best UMAP reducer to the embeddings,
        and saves a plot of the optimized embeddings.
        Random State can be toggled using config.py file.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(umap_objective, n_trials=100)
    best_params = study.best_params
    print("Best Parameters:", best_params)
    path = os.path.join(CONFIG.BASE_RESULTS, "person_embeddings_minilm.json")
    embeddings_dict = load_embeddings(path)
    embeddings = np.array(list(embeddings_dict.values()))

    best_reducer = umap.UMAP(**best_params, random_state=CONFIG.RANDOM_STATE)
    optimized_embedding_2d = best_reducer.fit_transform(embeddings)
    mmv.MatchMakingModel.save_plot_matches(
        optimized_embedding_2d,
        f"visualization_umap_optimised_{CONFIG.RANDOM_STATE}",
        embeddings_dict,
    )
