import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import mglearn
import yaml
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize 
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

    
def run_pipeline():
    '''run entire consultation nlp pipeline'''
    pass


'''Data Import'''

'''    
Project Details:
    
File pathway:"D:\nlp_pipeline_brenng\nlp_pipeline_brenng\nlp_projects\census_transf_2023"
File: 2023 consultation mock data.csv
Data Column: (Multiple columns) Start with...cens_test_1

row count (raw): In progress...TBC
'''


path_to_file = r"D:\nlp_pipeline_brenng\nlp_pipeline_brenng\nlp_projects\census_transf_2023\2023_consultation_mock_data.csv"
nlp_censtranf = pd.read_csv(path_to_file)

nlp_censtranf.shape
nlp_censtranf["cens_test_1"].head(5) #may need to change "" to '' for column/variable name!!!

'''Data Pre-processing'''


# lower casing

nlp_censtranf["cens_test_1"] = nlp_censtranf["cens_test_1"].str.lower()

# remove punctuation

print(string.punctuation)

def remove_punct(ptext):
    ptext = re.sub(string=ptext,
                   pattern="[{}]".format(string.punctuation), 
                   repl="")
    return ptext

nlp_censtranf["cens_test_1"] = nlp_censtranf["cens_test_1"].apply(remove_punct)


#Bad Starts...identify badstarts and update function below

nlp_cnestranf = ( 
    opn_coliv_nlp
    .loc[lambda df: ~df["cens_test_1"].str.startswith('9999')]
    .loc[lambda df: df["cens_test_1"]!='0']
)


nlp_censtranf["cens_test_1"].shape


# spelling mistakes...needs a confirmation, it's running but would benefit from another opinion

WORD = re.compile(r'\w+')
spell = SpellChecker()

def reTokenize(doc):
    tokens = WORD.findall(doc)
    return tokens

text = ["nlp_censtranf['cens_test_1']"]


def spell_correct(text):
    sptext =  [' '.join([spell.correction(w).lower() for w in reTokenize(doc)])  for doc in text]    
    return sptext    

print(spell_correct(text)) 


'''SOMETHING NOT RIGHT HERE''' #Remove Blanks...DOESN'T SEEM RIGHT AT ALL...should be well into double figures of blanks being removed

# replace Blank Cells by NaN in pandas DataFrame Using replace() Function

nlp_censtranf["cens_test_1"] = nlp_censtranf["cens_test_1"].replace(r'^s*$', float('NaN'), regex = True)  # Replace blanks by NaN
op.dropna(subset = ["cens_test_1"], inplace = True)     # Remove rows with NaN

nlp_censtranf["cens_test_1"].shape # not removing all due to spaces at the start of the blank row...look for a regex that looks for spaces...or look up 'strip' for same purose (don't ID a character and it will remove spaces)

'''START RENAMING CONVENTION FROM HERE?'''

'''#Cleaning data'''

#Tokenize and remove short tokens
#put in a text check here
nlp_censtranf["cens_test_1_tokens"] = nlp_censtranf["cens_test_1"].apply(nltk.word_tokenize)

 # remove tokens of 2 lenght 2 or less...amend as needed
 
def remove_short_tokens(ptokens):
    return [token for token in ptokens if len(token) > 2]

nlp_censtranf["cens_test_1_tokens"]  = nlp_censtranf["cens_test_1_tokens"] .apply(remove_short_tokens) #modify based on new columns approach


#Stem vs Lemm....USING STEMMING FOR NOW, but change to Lemming asap

# Define stemming function

def stemming(ptoken):
    # create stemming object
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in ptoken] 

# apply stemming
nlp_censtranf["cens_test_1_stemmed"] = nlp_censtranf["cens_test_1_tokens"] .apply(stemming)


#Stopwords

print(sw.words('english'))

def clean_stopwords(tokens):
    # define stopwords
    stop_words = set(sw.words('english'))
    # loop through each token and if the word isn't in the set 
    # of stopwords keep it
    return [item for item in tokens if item not in stop_words]

# can add an exclusion list code here

nlp_censtranf["cens_test_1_stopwords"] = nlp_censtranf["cens_test_1_stemmed"].apply(clean_stopwords)


'''EDA and Summary first looks'''
#REBUILD THIS SECTION ONCE OPN WORK UPDATED

'''#Topic Modelling'''
#what's not applied from cleaning above in TM below?

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

censtranf_respns = nlp_censtranf["cens_test_1"] #what's here and how is this utlising the pre-processing and data cleaning?
censtranf_respns = nlp_censtranf.reset_index(drop=True)

nlp_censtranf[2] #quick QA

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
