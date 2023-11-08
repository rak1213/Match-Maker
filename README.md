# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers 

1. all-MiniLM-L6-v2 model

    ![Sample output of script](Results/visualization_minilm.png?raw=true)

2. all-mpnet-base-v2 model

    ![Sample output of script](Results/visualization_mpnet.png?raw=true)

## What Are Embeddings?

In simple terms, embeddings are a way to translate words and sentences into a language that computers can understand. Each word has its own set of characteristics, meanings and usage in different contexts, and relationships with other words. Embeddings turn these characteristics into numbers that represent each word uniquely, much like assigning a specific code to each word based on its features.

Words with similar meanings has similar numerical representations, and are closer to each other in the digital space. Sentence is also not just a collection of words, but a pattern that they form together. Sentence embeddings look at this pattern to understand the sentence as a whole.

![alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JICAZM0gRjD9kSyMZtm99Q.png)

In the image we can see that the similar words are placed together like media and press, Obama and president and others.



## Data Analysis

For data analysis we selected 4 sentences from our dataset:

1. The below image shows the actual data entered by people

    ![Alt text](Images/actual_data.png?raw=true)


2. The below image shows the data modifed by us in order to perform data analysis and compare embeddings

    ![Alt text](Images/modified_data.png?raw=true)

Results: Cosine similarities
    
![Alt text](Results/similarity_embeddings.png?raw=true)


For Rakshit and Neeyati, we get a high cosine similarity score of more than 78%, which was expected too as the words in the sentences were only replaced by their synonyms as in case of sleeping with napping or paraphrased the whole sentence again as in case of Rakshit. It shows that transformer was capable of finding similarities between the words and was able to capture semantic relationships.

For Tejasvi, the sentence was kept as it is and we got a score of 100% which shows that embeddings created were same.

For Sylvester, we replaced the the word outdoor with inside and made the sentence completely opposite, but suprisingly got a score of 69%. This may be due to the reason, all the words in the sentence was same except one or two. It shows that irrespective of opposite meaning of one word, embeddings formed by were quite similar due to usage of same words in same context.



## Embedding Sensitivity Tests

The core of our comparison relied on Spearman’s rank correlation coefficient, which measures the similarity in the ordering of classmates from a particular person's embedding from closest to farthest, as determined by the two models. The Spearman’s rank correlation coefficient was found to be 0.81761 (81.761%), indicating a strong positive correlation between the rankings produced by the two models. This high correlation suggests that despite the change in model, the overall ordering of classmates in terms of similarity to a reference embedding (mine) remained relatively consistent.

It was observed that there were some notable shifts in rankings for certain individuals. Specifically, a few classmates who were ranked moderately far in the all-MiniLM-L6-v2 model appeared significantly closer in the all-mpnet-base-v2 model and vice versa. This shift underscores the nuanced differences in how each model processes and represents semantic similarities in text.

Below is the flattened embedding space of names clustered based on their interests using the sentence-transformers all-mpnet-base-v2 model.

![Sample output of script](Results/visualization_mpnet.png?raw=true)


## Dimension Reduction Analysis

We tried getting the visualizations at different random seeds like 0, 23 and 42 (shown on top) for sentence transformer all-MiniLM-L6-v2 model. Below are the visualizations as shown:

![Sample output of script](Results/visualization_minilm_rs_0.png?raw=true)

![Sample output of script](Results/visualization_minilm_rs_23.png?raw=true)

When using different random seed in UMAP dimension reduction exhibited notable fluctuations.This variability indicated a certain degree of instability or sensitivity in the model, suggesting that the initial embeddings' placement significantly influenced the resultant 2D projections.

However, post-tuning, UMAP with optimized parameters using optuna for n_neighbors, min_dist and metric, the model exhibited markedly reduced sensitivity to changes in the random seed. The visualizations remained more consistent across different seeds, indicating enhanced stability in the dimension reduction process by UMAP. This improvement suggests that the tuned parameters better captured the intrinsic structure of the data, making the model less reliant on the initial random configuration and more robust to variations in starting conditions.

### RESULTS:

1. Output for Tuned UMAP at ramdom seed 0

    ![Sample output of script](Results/visualization_umap_optimised_0.png?raw=true)

2. Output for Tuned UMAP at ramdom seed 23

    ![Sample output of script](Results/visualization_umap_optimised_23.png?raw=true)

3. Output for Tuned UMAP at ramdom seed 42

    ![Sample output of script](Results/visualization_umap_optimised_42.png?raw=true)
