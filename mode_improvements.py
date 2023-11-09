import optuna
import os
import numpy as np
import config as CONFIG
import match_making_visualizer as mmv
from umap_enhancement import load_embeddings
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.manifold import TSNE



def tsne_objective(trial):
    """
    Objective function for optimizing t-SNE hyperparameters using Optuna.

    Args:
        trial (optuna.trial.Trial): A trial instance from Optuna.

    Returns:
        float: The mean Spearman correlation coefficient across all pairs of embeddings.

    Note:
        This function suggests hyperparameters (perplexity, learning_rate, n_iter) for t-SNE,
        loads embeddings, applies dimensionality reduction, and calculates the mean Spearman
        correlation coefficient between cosine similarity ranks and Euclidean distance ranks of the embeddings.
    """
    perplexity = trial.suggest_int("perplexity", 5, 50)
    learning_rate = trial.suggest_float("learning_rate", 10, 1000)
    n_iter = trial.suggest_int("n_iter", 250, 5000)

    path = os.path.join(CONFIG.BASE_RESULTS, "person_embeddings_minilm.json")
    embeddings_dict = load_embeddings(path)  
    embeddings = np.array(list(embeddings_dict.values()))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=CONFIG.RANDOM_STATE
    )
    embedding_2d = tsne.fit_transform(embeddings)

    correlations = []
    for i in range(len(embeddings)):
        cos_similarities = np.array([1 - cosine(embeddings[i], emb) for emb in embeddings])
        euclidean_dists = np.linalg.norm(embedding_2d - embedding_2d[i], axis=1)

        cos_ranks = np.argsort(-cos_similarities)
        euclidean_ranks = np.argsort(euclidean_dists)

        correlation, _ = spearmanr(cos_ranks, euclidean_ranks)
        correlations.append(correlation)

    return np.mean(correlations)


if __name__ == "__main__":
    """
    Main execution block to optimize t-SNE hyperparameters and save a plot of the optimized embeddings.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(tsne_objective, n_trials=100)
    best_params = study.best_params
    print("Best Parameters:", best_params)

    path = os.path.join(CONFIG.BASE_RESULTS, "person_embeddings_minilm.json")
    embeddings_dict = load_embeddings(path)
    embeddings = np.array(list(embeddings_dict.values()))

    best_tsne = TSNE(**best_params, random_state=CONFIG.RANDOM_STATE)
    optimized_embedding_2d = best_tsne.fit_transform(embeddings)

    mmv.MatchMakingModel.save_plot_matches(
        optimized_embedding_2d,
        f"visualization_tsne_optimised_{CONFIG.RANDOM_STATE}",
        embeddings_dict
    )
