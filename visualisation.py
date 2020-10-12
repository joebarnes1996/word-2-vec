#==========================================================================

#==========================================================================
"""
1. Import packages
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


#==========================================================================

#==========================================================================
"""
2. Load and transform the data
"""
# set directory
os.chdir(r'C:\Users\joeba\github_projects\word2vec\embeddings')

words_df = pd.read_csv('words.csv', index_col=0)
vecs = np.loadtxt('vecs.csv', delimiter=',')


# standardise the vectors
std = StandardScaler()
vecs_std = std.fit_transform(vecs)

# perform PCA on the standardised data
pca = PCA()
data_red = pca.fit_transform(vecs_std) # reduced dimensionality data

eig_vals = pca.explained_variance_ratio_
eig_vecs = pca.components_


#==========================================================================

#==========================================================================
"""
3. Visualise the results
"""
# create a function to scatter words in principal component space,
# then highlight selected words
def plot(comp_1=0, comp_2=1, size=10, words_highlight=0, file_name=None, leg='best'):
    
    # scatter the words in the PC space
    plt.figure(figsize=(size, size))
    plt.grid()
    plt.scatter(data_red[:, comp_1], data_red[:, comp_2], marker='.', 
                s=size, c='gray')
    
    # highlight the selected words
    if words_highlight != 0:

        for i in words_highlight:
            
            # in case the word doesn't exist
            try:
            
                index = words_df[words_df.Words == i].index[0]
                plt.scatter(data_red[index, comp_1], data_red[index, comp_2], 
                            marker='.', s=size*40, label=i)

            except:
                
                pass
    
    # add information to plot
    plt.legend(fontsize=size*1.5, loc=leg) 
    plt.xlabel('PC{} ({:.2f}%)'.format(comp_1+1, 100*eig_vals[comp_1]), 
               fontsize=size*2)
    plt.ylabel('PC{} ({:.2f}%)'.format(comp_2+1, 100*eig_vals[comp_2]), 
               fontsize=size*2)
    
    # save plot
    try:
    
        os.chdir(r'C:\Users\joeba\github_projects\word2vec\images')
        plt.savefig(file_name)
        
    except:
        
        pass
    
    plt.show()


#============================
# try with colours
plot(words_highlight=['red', 'yellow', 'orange', 'black', 'white', 'green',
                      'blue', 'crimson', 'brown', 'gray'],
     file_name='colours')

# try with months
plot(words_highlight=['january', 'february', 'march', 'april', 'may',
                      'june', 'july', 'august', 'september',
                      'october', 'november', 'december'],
     file_name='months', leg='upper left')


# days
plot(words_highlight=['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                      'saturday', 'sunday'],
     file_name='months', leg='upper left')



# try with countries and cities
plot(words_highlight=['england', 'france', 'germany', 'italy',' china', 'india',
                      'poland', 'australia', 'america'],
     file_name='countries')


# try with random
plot(words_highlight=['sea', 'ocean', 'waves', 'shore', 'beach', 'sand', 
                      'deep', 'blue'],
     file_name='ocean')





# try with outliers
max_pc1 = words_df.Words.iloc[data_red[:,0].argmax()]
min_pc1 = words_df.Words.iloc[data_red[:,0].argmin()]
max_pc2 = words_df.Words.iloc[data_red[:,1].argmax()]
min_pc2 = words_df.Words.iloc[data_red[:,1].argmin()]

plot(words_highlight=[min_pc1, max_pc1, min_pc2, max_pc2],
     file_name='outliers')















