'''Step 1: Packages'''
#comment in usages later

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import wordcloud
import spacy
import os
import string
import re
import mglearn
from collections import Counter
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords as sw
nltk.data.path.append("../local_packages/nltk_data")
stopwords = sw.words("english")
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from IPython.display import display, HTML
import datetime

import scipy
from scipy.signal import savgol_filter
from spellchecker import SpellChecker

import xlrd
    
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import KeyedVectors
import gensim.downloader as api



'''Step 2: Data'''

path_to_file = "Z:\Covid\opn_coliv_nlp\OPN2304DK_Main_Income_nlp.csv"
opn_coliv_nlp = pd.read_csv(path_to_file)

opn_coliv_nlp.head(5)
opn_coliv_nlp.shape
opn_coliv_nlp['COL_OwnWords'].shape

'''Step 3: Functions'''

'''#Preprocessing''' #Following the method of amending the column throughout...amend if needed to create new columns

#sample check

opn_coliv_nlp['COL_OwnWords'][18]

# lower casing

opn_coliv_nlp['COL_OwnWords'] = opn_coliv_nlp['COL_OwnWords'].str.lower()

opn_coliv_nlp['COL_OwnWords'][18]

opn_coliv_nlp['COL_OwnWords'].shape


# remove punctuation

print(string.punctuation)

# Below is a function that uses regex to remove punctuation from strings
def remove_punct(ptext):
    # replace any punctuation with nothing "", effectively removing it
    ptext = re.sub(string=ptext,
                   pattern="[{}]".format(string.punctuation), 
                   repl="")
    return ptext

opn_coliv_nlp['COL_OwnWords'] = opn_coliv_nlp['COL_OwnWords'].apply(remove_punct)

opn_coliv_nlp['COL_OwnWords'][18]

opn_coliv_nlp['COL_OwnWords'].shape #should be seeing some lines removed here...there are some , and . starts. Might have become blanks...recheck count after blanks removed



#Bad Starts...9999 is the only one I can see in the table so far, which isn't removed through other methods (0 could be taken out of this function)

opn_coliv_nlp = ( 
    opn_coliv_nlp
    .loc[lambda df: ~df['COL_OwnWords'].str.startswith('9999')]
    .loc[lambda df: df['COL_OwnWords']!='0']
)


opn_coliv_nlp['COL_OwnWords'].shape


# spelling mistakes...needs a confirmation, it's running but would benefit from another opinion

WORD = re.compile(r'\w+')
spell = SpellChecker()

def reTokenize(doc):
    tokens = WORD.findall(doc)
    return tokens

text = ["opn_coliv_nlp['COL_OwnWords']"]

def spell_correct(text):
    sptext =  [' '.join([spell.correction(w).lower() for w in reTokenize(doc)])  for doc in text]    
    return sptext    

print(spell_correct(text)) 

opn_coliv_nlp['COL_OwnWords'][18] #find a row with an error to confirm this function

opn_coliv_nlp['COL_OwnWords'].shape



'''SOMETHING NOT RIGHT HERE''' #Remove Blanks...DOESN'T SEEM RIGHT AT ALL...should be well into double figures of blanks being removed

# replace Blank Cells by NaN in pandas DataFrame Using replace() Function

opn_coliv_nlp['COL_OwnWords'] = opn_coliv_nlp['COL_OwnWords'].replace(r'^s*$', float('NaN'), regex = True)  # Replace blanks by NaN
opn_coliv_nlp.dropna(subset = ['COL_OwnWords'], inplace = True)     # Remove rows with NaN

opn_coliv_nlp['COL_OwnWords'].shape # not removing all due to spaces at the start of the blank row...look for a regex that looks for spaces...or look up 'strip' for same purose (don't ID a character and it will remove spaces)

'''START RENAMING CONVENTION FROM HERE?'''

'''#Cleaning data'''

#Tokenize and remove short tokens

opn_coliv_nlp['COL_OwnWords'][456]
opn_coliv_nlp['COL_OwnWords_tokens'] = opn_coliv_nlp['COL_OwnWords'].apply(nltk.word_tokenize)
opn_coliv_nlp['COL_OwnWords_tokens'][18] #not outputting tokenized version as it should

opn_coliv_nlp['COL_OwnWords_tokens'].shape


 # remove tokens of 2 lenght 2 or less
'''need to modify when I correct for the new columns approach'''
 
def remove_short_tokens(ptokens):
    return [token for token in ptokens if len(token) > 2]

opn_coliv_nlp['COL_OwnWords_tokens'] = opn_coliv_nlp['COL_OwnWords_tokens'].apply(remove_short_tokens) #modify based on new columns approach

opn_coliv_nlp['COL_OwnWords_tokens'][18]

#Stem vs Lemm....USING STEMMING FOR NOW, but change to Lemming asap

# Define stemming function

def stemming(ptoken):
    # create stemming object
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in ptoken] 

# apply stemming
opn_coliv_nlp['COL_OwnWords_stemmed'] = opn_coliv_nlp['COL_OwnWords_tokens'].apply(stemming)

# tokens post-stemming
opn_coliv_nlp['COL_OwnWords_stemmed'][18]
# note: chose [10] as a good example, but my reindexing isn't working at the moment so [10] will change

#Stopwords

print(sw.words('english'))

def clean_stopwords(tokens):
    # define stopwords
    stop_words = set(sw.words('english'))
    # loop through each token and if the word isn't in the set 
    # of stopwords keep it
    return [item for item in tokens if item not in stop_words]

# can add an exclusion list code here

opn_coliv_nlp['COL_OwnWords_stopwords'] = opn_coliv_nlp['COL_OwnWords_stemmed'].apply(clean_stopwords)

opn_coliv_nlp['COL_OwnWords_stopwords'][18] 


'''EDA and Summary first looks'''

#BOW to Word Cloud#

#eda and coutervectorization???

opn_coliv_nlp['COL_OwnWords_eda'] = opn_coliv_nlp['COL_OwnWords_stopwords'] 

opn_coliv_nlp['COL_OwnWords_eda'][18] #renamed to eda to mark the next phase of analysis


#join tokens and the create work an assessment of the vocabulary (frequency etc)
def join_tokens(tokens):
    return ' '.join(tokens)

opn_coliv_nlp['COL_OwnWords_str'] = opn_coliv_nlp['COL_OwnWords_eda'].apply(join_tokens)

coliv_words = opn_coliv_nlp['COL_OwnWords_str']

#CountVectorizer is a transformer...is it used to fit to my data and then start the tokenization process
vect = CountVectorizer()
vect.fit(coliv_words)

#the the vocabulary can be built and displayed...using the vocabulary_ attribute
print("Vocabulary size: {}".format(len(vect.vocabulary_)))

print("Vocabulary content:\n{}".format(vect.vocabulary_))

#1 word frequency count

# Tokenise in a basic manner
#REMOVED TOKENIZING HERE BECAUSERE THERE's NO LOGIV TO IT...REMOVE THIS LINE WHEN CONFIDENT

#1 word frequency count

# We want one full list of tokens to be analysed...can be applied directly to the column/variable??
coliv_freq = []
for item in opn_coliv_nlp['COL_OwnWords_eda'] :
    # extend is similar to append, but combines lists into
    # one larger list
    coliv_freq.extend(item)

# A quick look at our frequency dictionary...different from above?
counter = Counter(coliv_freq)
print(counter)

# calling .most_common() on our Counter
# object will sort them for us
counter.most_common(25)

# separate out the tokens and counts into lists
tokens, counts = zip(*counter.most_common())

def plotall(px, py):
    
    plt.xticks(fontsize=12, rotation=90)
    plt.ylabel('Frequency')
    plt.xlabel("Tokens")
    plt.bar(px, py)
    plt.show()
    
# select only the top 10 of each
plotall(tokens[:10], counts[:10])

# Join all the text data
text = " ".join(coliv_freq)


#Wordcloud

# The text string is then passed to the wordcloud function:
wordcloud = WordCloud(max_font_size=50, 
                      max_words=100, 
                      background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


'''#Topic Modelling'''


# step 1 fit the data...this is more of a data prep stage, takes the top 15% most occuring words away, up to 10k words (I have only 1070)
# resetting to my total number of rows...742

vect = CountVectorizer(max_features=10000, max_df=.15)
coliv_wordsbows  = vect.fit_transform(coliv_words)  #creates the bow from vect

coliv_wordsbows 

# play with topic number here and n_words below to get a useful output and tune the model...what will be the logic?

# testing a range of 4-20 topics to get a better fit (o vs u)

lda5 = LatentDirichletAllocation(n_components=5, learning_method="batch",
                              max_iter=25, random_state=0)

document_topics5 = lda5.fit_transform(coliv_wordsbows)

# then view a selection of the 100 topics...increased the words to 20
# I'm playing with the topic and word amounts below

topics = np.array([0,1,2,3,4])

sorting = np.argsort(lda5.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names, 
                          sorting=sorting, topics_per_chunk=5, n_words=10)

# later...change this to columns so the data extraction wouldn't run through the indices first it would be run through the data rather than the indices

document_topics5 #provides the weights???

coliv_respns = opn_coliv_nlp['COL_OwnWords'] #what's here and how is this utlising the pre-processing and data cleaning?
coliv_respns = coliv_respns.reset_index(drop=True)

coliv_respns[2] #quick QA

# running a loop to repeat...the above steps??

def topic_summary(topic_number): #this brings in the supporting statements with the topic (original statements so uncleaned is perfect)
    

    topics = [topic_number]
    mglearn.tools.print_topics(topics=topics, feature_names=feature_names, 
                          sorting=sorting, topics_per_chunk=5, n_words=10)

    responses = np.argsort(document_topics5[:, topic_number])[::-1]

    for i in responses[:5]:
        print(coliv_respns[i],".\n")
        
for i in range(5):
    topic_summary(i)
       
#plot the topics...with original names (top two words)
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
topic_names = ["{:>2} ".format(i) + " ".join(words)
               for i, words in enumerate(feature_names[sorting[:, :2]])]

ax.barh(np.arange(5), np.sum(document_topics5, axis=0))
ax.set_yticks(np.arange(5))
ax.set_yticklabels(topic_names, ha="left", va="top")
ax.invert_yaxis()
ax.set_xlim(0, 300)
yax = ax.get_yaxis()
yax.set_tick_params(pad=130)
plt.tight_layout()

#then insert topic labels
# my topic labels

topic_labels = ["The first label", 
                "The second label", 
                "The second label", 
                "The third label", 
                "The fourth label"]



fig, ax = plt.subplots(1, 1, figsize=(10, 8))
topic_names = ["{:>2} {}".format(i, label)
               for i, label in enumerate(topic_labels)]

ax.barh(np.arange(5), np.mean(document_topics5, axis=0))
ax.set_yticks(np.arange(5))
ax.set_yticklabels(topic_names, ha="right", va="center")
ax.invert_yaxis()
ax.set_xlim(0, 0.5)
yax = ax.get_yaxis()
yax.set_tick_params(pad=10)
plt.tight_layout()
'''end for now'''




'''#Splitting and Time Series another time'''
#age
#gender
#deprivation
#tenure (e.g. renters vs owners)
