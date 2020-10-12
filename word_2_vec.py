# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:33:13 2020

@author: joeba
"""


#==========================================================================

#==========================================================================
"""
Using the following resources:
    
    gensim package: https://blog.cambridgespark.com/tutorial-build-your-own-embedding-and-use-it-in-a-neural-network-e9cde4a81296
    
    downloading data: http://www.gutenberg.org/

    Visualising word projections: http://projector.tensorflow.org/
    
"""


#==========================================================================

#==========================================================================
"""
1. Import packages
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
import multiprocessing
import time



#==========================================================================

#==========================================================================
"""
2. Load data

Import all of the literature text files as strings, then join all the 
strings together.
"""

# change directory to the location of the various literature .txt files
path = r'C:\Users\joeba\github_projects\word2vec\data'
os.chdir(path)

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


# read the data
data_raw = ''

for i in files:

    data_add = open('{}'.format(i)).read()
    data_add = data_add.replace('\n', ' ')
    
    data_raw += data_add


#==========================================================================

#==========================================================================
"""
3. Pre-process the data

For simplicty, only use the characters of space, full stop, and any alphabetical
characters. Also ensure that all characters are lower case.

Then split the text into sentences, which are lists of words, so that they 
are easily digested by the Word2Vec algorithm

"""

#======================================
# remove any special characters from the text
def standardise_text(raw_text, allowed_chars, replacement_char= ' '):
    
    # make all characters lowercase
    raw_text = raw_text.lower()
    
    # replace any characters that arent part of our list (allowedChars) 
    # using a replacement of 'replacementChar'
    standardised_text = ''
    
    for char in raw_text:
        
        if allowed_chars.find(char) == -1: # char isn't in the allowed list
            
            standardised_text = standardised_text + replacement_char
            
        else:
            
            standardised_text = standardised_text + char
            
    return standardised_text


# split and standardise data to be individual sentences with no punctuation
data_std = standardise_text(data_raw, ' .abcdefghijklmnopqrstuvwxyz', ' ')
sentences = data_std.split('.')

# split each sentence by word
sentences_std = []

for i in range(len(sentences)):
    
    sentences_std.append(sentences[i].split())
    
    print(100 * i / len(sentences_std))





#======================================

# delete any sentences less than 5 words
def delete_short_sequences(sequences, window_size=5):
    
    sequences_not_short = sequences.copy()
    
    # get indices of sequences shorter than window_size
    indices = []
    
    for i in range(len(sequences_not_short)):
        
        if len(sequences_not_short[i]) < window_size:
            
            indices.append(i)
            
    # sort indices in descending order
    indices.sort(reverse=True)
    
    # delete all sequences according to indices
    for i in indices:
        
        sequences_not_short.pop(i)
        
    return sequences_not_short

# delete short sentences
sentences = delete_short_sequences(sentences_std)

print('Pre-proc complete')

#==========================================================================

#==========================================================================
"""
4. Create the embedding model
"""
# set dimensions for embedded vectors (number of nodes in hidden layer)
embed_dim = 300

# create the model
start = time.time()
word_2_vec = Word2Vec(sentences, size=embed_dim, window=5, min_count=5,
                      negative=15, iter=10, 
                      workers=multiprocessing.cpu_count())

print('Took {:.3f} seconds'.format(time.time() - start))

print('Model trained')



#==========================================================================

#==========================================================================
"""
4. Evaluate embeddings

Look at similar words to other words, based on the cosine of the angle between
two word vectors

"""
# get the word vectors
word_vectors = word_2_vec.wv


# create a function to find simliar words
def similar_words(word=0):
    
    # if no word, it will prompt for one
    if word == 0:
    
        word = input('Enter word to compare: ')
        word = word.lower()

    # store a list of similar words, and their angles with a word
    similar_words = []
    cos_theta  = []
    
    # get the gensim format of similar words
    similar_words_output = word_vectors.similar_by_word(word)
    
    # create lists of similar words and their cos thetas
    for i in range(len(similar_words_output)):
        
        similar_words.append(similar_words_output[i][0])
        cos_theta.append(similar_words_output[i][1])
    
    # store in dataframe
    similarities_df = pd.DataFrame({'Words' : similar_words,
                                    'Cos_theta': cos_theta})

    print('\n\n\n') 
    print('The most similar words to {} are: \n'.format(word))
    
    print(similarities_df)
    

# print a few examples
similar_words('december')
similar_words('time')
similar_words('gun')
similar_words('crime')
similar_words('london')




#==========================================================================

#==========================================================================
"""
5. Store words and their vectors
"""

# get lists of words and corresponding vectors
words = list(word_vectors.vocab.keys())
vecs  = []

for i in range(len(words)):
    
    vecs.append(word_vectors.get_vector(words[i]))



# save words and vectors
words_df = pd.DataFrame(words, columns=['Words'])
vecs = np.array(vecs)

os.chdir(r'C:\Users\joeba\github_projects\word2vec\embeddings')
words_df.to_csv('words.csv')
np.savetxt('vecs.csv', vecs, delimiter=',')





#==========================================================================

#==========================================================================
"""
6. Assess symmetry between any two words
"""

def compare_words(word1=0, word2=0, return_val=False):
    
    # input the words
    if word1 == 0 or word2 == 0:
        
        word1 = input('Give first word: ')
        print('\n')
        word2 = input('Give second word: ')
        print('\n')
    
    # convert to lower case
    word1 = word1.lower()
    word2 = word2.lower()
    
    try:
        
        # get the vectors for each word
        vec1 = vecs[words_df[words_df.Words == word1].index[0]]
        vec2 = vecs[words_df[words_df.Words == word2].index[0]]
        
        
        # calculate cos(theta) between the two word vectors
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # state similarity
        print('{} and {} have similarity score of: \n\n {:.3f}'.
              format(word1, word2, cos_theta))
        
        # if needed, return the value
        if return_val == True:
            
            return cos_theta

    # if a word is incorrec
    except:
        
        print('One or more words are incorrect.')
    

# test it works
compare_words('Paris', 'London')
compare_words('France', 'England')
compare_words('King', 'Queen')
compare_words('King', 'peasant')





#==========================================================================

#==========================================================================
"""
7. Create function to assess the vector differences

will perform an operation of the format word1 - word2 + word3

"""

def vec_diff(num_similar=5):
    
    # prompt inputs for each word / operation
    word1 = input('Input word1: ')
    word2 = input('Input word2: ')
    word3 = input('Input word3: ')
    
    # get the word vectors    
    vec1 = vecs[words_df[words_df.Words == word1].index[0]]
    vec2 = vecs[words_df[words_df.Words == word2].index[0]]
    vec3 = vecs[words_df[words_df.Words == word3].index[0]]
    
    # get the difference vector
    vec = vec1 - vec2 + vec3
    
    # similarities
    sims = []
    
    for i in range(len(vecs)):
        
        sims.append(np.dot(vec, vecs[i]) / (np.linalg.norm(vec) * np.linalg.norm(vecs[i])))
        
        print('Comparing: {:.1f}%'.format(100 * i / len(vecs)))
    
    # put into a dataframe and get the n most similar
    sims_df = pd.DataFrame(sims, columns=['Similarities']).sort_values(by='Similarities',
                                                                       ascending=False)
    n_most_sim = sims_df[:num_similar]
    
    # get list of most similar vectors and their index
    ind_most_sim = list(n_most_sim.index)
    sim_most_sim = list(n_most_sim.Similarities)
    
    # convert indices to words
    words_most_sim = []
    
    for i in ind_most_sim:
        
        words_most_sim.append(words_df.Words.iloc[i])
        
    # convert into a dataframe
    sim_words_df = pd.DataFrame({'Words':words_most_sim,
                                 'Cos_theta':sim_most_sim})
    
    # display the results
    print('The most similar words to {} - {} + {} are: \n\n{}'
          .format(word1, word2, word3, sim_words_df))
        
    


"""

Create some PCA visualisations.

"""













