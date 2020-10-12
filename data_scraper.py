#==========================================================================

#==========================================================================
"""
1. Import packages
"""
import urllib.request as urllib2 
import csv
from bs4 import BeautifulSoup
import os




#==========================================================================

#==========================================================================
"""
2. Create function to download data
"""
# set the directory to save data to
os.chdir(r'C:\Users\joeba\github_projects\word2vec\data')


# settings
hdr = {'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'User-Agent' : 'Magic Browser'
       }


# create a function to download book texts
def book_scraper(no_books=50, random=False, start_from=1111):
    
    # set a counter
    count = 1
    
    for i in range(start_from, start_from + no_books):
        
        try:
            
            # if would rather a random number...
            if random == True:
                
                i = np.random.randint(60000)
            
            # state the link
            link = 'http://www.gutenberg.org/cache/epub/{}/pg{}.txt'.format(i, i)
            
            # load the webpage
            page = urllib2.Request(link, headers=hdr)
            content = urllib2.urlopen(page).read()
            
            # get the soup
            soup = BeautifulSoup(content, 
                             'html.parser'
                             )
            
            # extract the text
            text = soup.get_text()
            
            # save the text file    
            f = open('{}.txt'.format(count), "a")
            f.write(text)
            f.close()
            
            
            # print progress
            print('{:.1f}% complete'.format(100 * count / no_books))
            
            # update the counter
            count += 1
            
        # if bad html link, then skip iteration
        except:
            
            pass





#==========================================================================

#==========================================================================
"""
3. Download data


Also, I advise setting random=True, as there are a number of CIA fact books
in the early publications, which is not entirely representative of the English
language.

"""

# download books
book_scraper(no_books=150)







