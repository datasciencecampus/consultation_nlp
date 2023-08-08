# User guide

This is the user guide for the `consultation-nlp-2023` project.

```{toctree}
:maxdepth: 2
./loading_environment_variables.md
```

## How to set the configure the model
The majority of the model configuration happens in the `question_model_config.yaml`

Within this file you will have configuration options for each of the questions that get's processed.

**example:**
```yaml
qu_12:
  max_features: null
  ngram_range: !!python/tuple [1,2]
  min_df: 2
  max_df: 0.9
  n_topics: 3
  n_top_words: 10
  max_iter:
    lda: 25
    nmf: 1000
  lowercase: true
  topic_labels:
    lda: null
    nmf:
        - "Admin Data"
        - "Research"
        - "Policy"
```
In this example you can see that the yaml file is indented at various levels.

### qu_12
type:str
At the top level of indentation, we have the question-id, in this case 'qu_12'. Each number corosponds to the column nuber of the raw input data (i.e. qu_12 is column 12 of the raw data csv).

### max_features
type: int (or null)
This is an optional value, which can either be null (which will convert to None when transposed to Python) or an integer value for the maximum number of text features to include.

### ngram_range
type: tuple (but looks like a bit like a list)
ngrams or word combination ranges, can help to increase the number of features you have in your dataset which is useful if multi-word phrases like "admin data" utilised a lot in the responses. The two values `[1,2]` corrospond to the start and end of the range. So this example would include unigrams (individual words) and bi-grams (2 word combinations). To have only one word combinations, you can change the settings to `[1,1]`. You can also include tri-grams and longer if you wish.

### min_df
type: int or float
This is a way of filtering out less important words, that don't appear in enough responses.  `min_df` can either be a float value (e.g. 0.1), in which case it will be interpreted as a proportion, or an integer value (e.g 1) where it will be interpretted as a number of responses.
So 0.1 would mean that a word needs to appear in at least 10% of the corpus to get through, or 2 would mean that it needs to appear in at least 2 documents.

### max_df
type: int or float
Similar to min_df, max_df is a way of filtering out words, but this time the more common words. This field also takes, floats and integers, interpretting them as proportions and absolute numbers respectively. So 0.9 would stop words appearing in more than 90% of documents from making their way through, or 100 would stop words that appear in more than 100 documents coming through.

### n_topics
type: int
This is the number of topics to attempt to model in the topic modelling, it must be an integer value.

### n_top_words
type: int
This is the number of top words to include in the modelling, it must be an integer value.

### max_iter
type: dictionary
This option breaks down further into `lda` and `nmf` which are both integers. This setting relates to the number of iterations for the models to run through in order to move towards convergence. You may need to adjust these seperately depending on model performance.

### lowercase
type: boolean
A switch setting for parsing words as lowercase or leaving them in their unadjusted form.

### topic_labels
type: dictionary
Again this one breaks down furhter into lda, and nmf, as it is likely that after you have run the models, you may wish to add specific topic lables for the plots you are generating. These can either be null or a list of strings. If you are setting labels, you must ensure there are the same number of labels as there are n_topics, otherwise the system will through an error.
