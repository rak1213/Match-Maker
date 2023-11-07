# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model.

![Sample output of script](Results/visualization.png?raw=true)



### What Are Embeddings?

In simple terms, embeddings are a way to translate words and sentences into a language that computers can understand. Each word has its own set of characteristics, meanings and usage in different contexts, and relationships with other words. Embeddings turn these characteristics into numbers that represent each word uniquely, much like assigning a specific code to each word based on its features.

Words with similar meanings has similar numerical representations, and are closer to each other in the digital space. Sentence is also not just a collection of words, but a pattern that they form together. Sentence embeddings look at this pattern to understand the sentence as a whole.

![alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*JICAZM0gRjD9kSyMZtm99Q.png)

In the image we can see that the similar words are placed together like media and press, Obama and president and others.


### Data Analysis

For data analysis we selected 4 sentences from our dataset:

1. The below image shows the actual data entered by people

    ![Alt text](Images/actual_data.png?raw=true)


2. The below image shows the data modifed by us in order to perform data analysis and compare embeddings

    ![Alt text](Images/modified_data.png?raw=true)

Results: Cosine similarity for it
    
![Alt text](Results/similarity_embeddings.png?raw=true)



For Rakshit and Neeyati, we get a high cosine similarity score of more than 78%, which was expected too as the words in the sentences were only replaced by their synonyms as in case of sleeping with napping or paraphrased the whole sentence again as in case of Rakshit. It shows that transformer was capable of finding similarities between the words and was able to capture semantic relationships.

For Tejasvi, the sentence was kept as it is an we got a score of 100% which shows that embeddings created were same.

For Sylvester, we replaced the the word outdoor with inside and made the sentence completely opposite, but suprisingly got a score of 69%. This may be due to the reason, all the words in the sentence was same except one or two. It shows that irrespective of opposite meaning of one word, embeddings formed by were quite similar due to usage of same words in same context.

