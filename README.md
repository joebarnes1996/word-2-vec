# Word2Vec Implementation

## Repository Overview

This repository demonstrates using the Gensim package with Python in order to create efficient word embeddings, and is trained on literature downloaded from http://www.gutenberg.org/. 

This repository contains the following three scripts:

* data_scraper.py - scrapes the gutenberg project website for text files of various literature, used to train the model.
* word_2_vec.py - pre-processing the data, then fits the Gensim Word2Vec model to the data.
* visualisation.py - reduces the data to the first two principal components, and shows groups of similar words to cluster together.




## Theory

In their initial string format, words cannot be easily digested by machine learning algorithms. While one can represent words by one-hot encoded vectors, though this has two major downfalls. One-hot vectors, by construction, are orthogonal to one another, hence being uncorrelated and not representing any similarities between words. As well as this, the dimensionality of these vectors is required to be equal to the number of words in a vocabulary, which is hundreds of thousands in the English language.

Much work has been done on word embeddings, however in 2013 Mikolov and his research team at Google developed a way to embed words into continuous vectors with significantly lower dimensions that the number of words in a set vocabulary. With their publication (in the literatire folder), they presented two models: skip-gram, and Continuous Bag of Words (CBOW). This project employs the skip-gram methodology, which I shall briefly describe below.

The skip-gram model is a neural network whose input is a word in a sentence, and whose output is the words which surround the input in a sentence, which is commonly referred to as the context. While training the neural network, the input word is represented as a one-hot vector, and the context is represented as the sum of one-hot vectors for each word. For example, if I took the sentence 'the dog went for a walk', one possible input, output pair could be ('dog', ('the', 'went', 'for', 'a', 'walk')). If no other words existed, 'dog' would be represented by [1, 0, 0, 0, 0, 0], while ('the', 'went', 'for', 'a', 'walk') would be represented by [0, 1, 1, 1, 1, 1]. Both the input and output have a dimensionality equal to the number of words in a vocabulary. The input is then passed to a lower dimensional hidden layer, and then passed through to the output layer, with a softmax distribution over the word space. The error is then calculated between predicted and real context, and the weights of the network are updated as per usual. Essentially, the model optimises the weights to find which words are linked to which context, hence finding links between words.

Given that only one element of a one-hot vector is non-zero, each word in a vocabulary can be represented as a row of the weights matrix between the input and the hidden layer. The rows of this matrix are thus the vector representations of the corresponding words in lower dimensional space.


## Results

Using word_2_vec.py, one can inspect the similar words to a given word, as well as investigate the differences between pairs of words, as well as basic word arithmetic (e.g. king - man + woman = queen).

The script visualistion.py shows the word vectors reduced to two dimensional space, according to principal component analysis (PCA). One can see in the below image that words with similar semantics, in this case things relating to the sea, cluster together.


![](ocean.png?raw=true "Optional Title")



