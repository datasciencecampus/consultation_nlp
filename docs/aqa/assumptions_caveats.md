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

## Spell Checker

The spell checker function identifies any words that it thinks are mis-spelled with a flag which then uses a Levenshtien Distance algorithm to find permutations within an edit distance of 2 from the original word. Each word within this list has a frequency value associated with it, the algorithm then finds the most likely word and replaces the mis-spelled word with it. A more thorough explaination of this method can be found in [Peter Norvig's Blog](https://norvig.com/spell-correct.html).

One of the potential challenges of using this method is that it can auto-correct words or phrases which are unknown to the pre-defined dictionary (e.g. DfE) or fail to adapt to words which are more or less likely in a specific context e.g. amin data -> main data, when it most probably is refering to admin data.

There are ways for us to override the spelling corrector. We have added a section called business_terminology in the config.yaml file, which allows us to add new words, or override existing word frequencies, so that some words are more likely to come out on top. But on average, the spell checker works correctly 70% of the time, according to Norvigs article.
