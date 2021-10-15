import pandas as pd
import numpy as np

class TwitterData:

    def __init__(self, filename, test_set=False):
        self.name = filename
        self.df = pd.read_csv(filename)
        new_col_values = self.get_originality()
        self.df['doc_freq'] = new_col_values[0]
        self.df['ln_word_cnt'] = new_col_values[1]
        popularity = self.get_popularity()
        self.df['keyword_pop'] = popularity[0]
        self.df['location_pop'] = popularity[1]
        
        # drop rows if not a test set
        if (test_set == False):
            self.df = self.df[self.df['keyword_pop'] != -1]
            self.df = self.df[self.df['location_pop'] != -1]
            self.labels = np.array(self.df['target'])   
        else:
            self.labels = np.zeros(len(self.df))
        
        # make the features in the df as a matrix
        self.asmatrix = self.get_matrix()
        
    
    # find the originality
    def get_originality(self):
        values = count_text(self.df['text'])
        return values
    
    # Get popularity lists for keyword and location columns
    def get_popularity(self):
        keyword_pop = [count_all_tweets(str(i), self.df['keyword']) for i in self.df['keyword']]
        location_pop = [count_all_tweets(str(i), self.df['location']) for i in self.df['location']]
        return keyword_pop, location_pop
    
    def get_matrix(self):
        matrix = np.zeros((len(self.df), 4))
        matrix[:, 0] = np.array(self.df['doc_freq'])
        matrix[:, 1] = np.array(self.df['ln_word_cnt'])
        matrix[:, 2] = np.array(self.df['keyword_pop'])
        matrix[:, 3] = np.array(self.df['location_pop'])
        
        return matrix

    
def count_text(text):
    origin = []
    word_lncounts = []
    for i in text:
        origin.append(count_all_tweets(i, text))
        word_lncounts.append(np.log(len(i.split())))
    return origin, word_lncounts
    

def count_all_tweets(tweet, text):
    count_list = []
    # number of tweets
    n = len(text)
    # check every word in the tweet
    if tweet == 'nan':
        return -1
    for word in tweet.split():
        # check if word is a common word
        if word in 'the a an and'.split():
            continue
        count = 0
        # count the occurances in the whole dataframe
        for t in text:
            if word in str(t).split():
                count += 1
                
        count_list.append(count / n)
        
    return np.sqrt(np.sum(count_list))

    