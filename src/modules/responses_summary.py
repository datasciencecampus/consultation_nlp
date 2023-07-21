# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:25:29 2023

@author: brenng
"""

'''
#the original code
def topic_summary(topic_number): #this brings in the supporting statements with the topic (original statements so uncleaned is perfect)
    

    topics = [topic_number]
    mglearn.tools.print_topics(topics=topics, feature_names=feature_names, 
                          sorting=sorting, topics_per_chunk=5, n_words=10)

    responses = np.argsort(document_topics5[:, topic_number])[::-1]

    for i in responses[:5]:
        print(coliv_respns[i],".\n")
        
for i in range(5):
    topic_summary(i)
       
    '''

'''refactoring for the census consultation'''

from numpy import ndarray

def topic_responses(n_topics:int, feature_names:str, sorting:ndarray-> str, topics_per_chunk:int, n_words:int, n_responses:int)->None: 
  
    """this prints the topic and top X words, plus the top X number of raw responses
    Parameters
    ----------
    topic_number:int #n_topics???
        X number of topics used in the model    
    feature_names:str
        labels used to name each topic
    sorting:??
        arranges the topics along one horizontal line
    topics_per_chunk:int
        X number of topics per used in the model #how is this diff from topic_number?
    n_words: int            
        X number of words used to create the topic
    n_responses:int
        Top X number of raw responses in the topic
        Returns
    -------
    None (prints to console)""" #change this to output to a word document

for topic in range (n_topics):
    mglearn.tools.print_topics(topics=topic, feature_names=feature_names, 
                          sorting=sorting, topics_per_chunk=5, n_words=10)

    responses = np.argsort(document_topics5[:, topic_number])[::-1]

    for i in responses[:10]:
        print(coliv_respns[i],".\n")
        
for i in range(5): #this is better placed within the for loop then remove the topics per chunk feature
    topic_summary(i)
       
    
    