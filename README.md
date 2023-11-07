# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model.

![Sample output of script](https://github.com/rak1213/matchmaking/blob/main/visualization.png?raw=true)



### What Are Embeddings?

In simple terms, embeddings are a way to translate words and sentences into a language that computers can understand. Each word has its own set of characteristics, meanings and usage in different contexts, and relationships with other words. Embeddings turn these characteristics into numbers that represent each word uniquely, much like assigning a specific code to each word based on its features.

Words with similar meanings has similar numerical representations, and are closer to each other in the digital space. Sentence is also not just a collection of words, but a pattern that they form together. Sentence embeddings look at this pattern to understand the sentence as a whole.

![alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JICAZM0gRjD9kSyMZtm99Q.png)

In the image we can see that the similar words are placed together like media and press, Obama and president and others.