# Assumptions and caveats log

This log contains a list of assumptions and caveats used in this analysis.

## NLTK stopwords
Stopwords are commonly used words which on their own don't really mean much. The NLTK package has a pre-defined list of stopwords which we have implemented in this pipeline, so we can focus our analysis on the key words that we think are likely to reveal more insights.

For transparency, here is a list of the NLTK stopwords:

`['when', "you'll", 'my', 'their', 'now', 'while', 'very', 'that', 'does', 'this', 'himself', 'nor', 'should', 'from', 'couldn', 'these', 'and', 'own', 'theirs', 'because', "haven't", "hadn't", 'hasn', 't', "wouldn't", 'has', "don't",
'at', 'above', 'between', 'is', 'weren', 'm', 'hadn', 'no', "she's", 'for', 'off', 'only', 'were', 'her', "doesn't",
'out', 'i', 'am', 'are', 'mightn', 'up', 'do', 'until', "won't", "it's", 'but', 'didn', 'ourselves', 'than', 'mustn', 'yours', "that'll", 'myself', "didn't", 'had', 'doing', 'yourselves', 'into', "you're", 'haven', 'ain', 'having', 'too', "mustn't", "needn't", "mightn't", 'doesn', 'a', 'before', 'further', 'by', 'most', 'any', 'whom', 'it', "isn't", 'they', 'will', 'he', 's', 'themselves', 'other', 'isn', 'all', 'to', 'hers', 'few', 'with', 'itself', 've', "aren't", 'shan', 'what', 'who', 'can', 'our', 'y', 'those', 'each', "you'd", 'if', 'did', 'his', "shan't", 'an', 'll', "couldn't", 'you', 'as', "should've", 'again', 'so', 'about', 'through', "shouldn't", 'him', 'more', 'have', 'once', 'your', 'how', 'there', 'just', 'd', 'needn', "wasn't", 'wouldn', 'or', 'down', "hasn't", "weren't", 'been', 'yourself',
'not', 'on', 'shouldn', 'ours', 'be', 'me', 'we', 'here', 'o', 'was', 'herself', 'after', 'aren', 'the', 'ma', 'which', "you've", 'then', 'against', 'same', 'being', 'below', 'in', 'wasn', 'over', 'don', 'them', 'both', 'some', 'such', 'during', 'why', 'its', 're', 'won', 'where', 'of', 'under', 'she']`

We have also added a few additional words which can be found in the config (e.g. 'census', 'data')
