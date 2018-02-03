
# coding: utf-8

# # Text Pre Process Pipeline
# 
# ## A quick example that can be easily applied to dataframes in machine learning problems using Python V3.0

# In[39]:


# Import necessary packages

import numpy as np
import pandas as pd
from scipy import sparse as sp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[40]:


# Import Natural language Toolkit example data and 'stopwords' set

from nltk.corpus import movie_reviews
from nltk.corpus import stopwords


# ### Get the _Movie Reviews_ Data into a Pandas DataFrame

# In[41]:


docs = [(str(movie_reviews.raw(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


# In[42]:


reviews = pd.DataFrame(docs)
reviews.columns=('X','y')

# The Category of a movie review is initially 'neg' or 'pos', changing here to 0 and 1, respectively

bin_encoder=LabelEncoder()
reviews.y=bin_encoder.fit_transform(reviews.y)


# In[43]:


reviews.head(5)


# ### Edit the _stopwords_ to include more words
# #### (This might be useful to filter out certain words you don't want included, but may not necessarily be in the default 'stopwords' list)

# In[44]:


mystopwords = (stopwords.words())
custom_stopwords = ('the','an','a','my','0','''''','!','nt','?','??','?!','%','&','UTC','(UTC)')


# ## Text Pre Process Pipeline
# 
# + ** 'count vectorizer' ** : Transformation from sentences to all lower-case words, stopwords removed, vectorized
# + ** 'chi2score' ** : Transformation that selects top k features related to the target based on ChiSquare test statistics
# + ** 'tf_transformer' ** : Transformation that transforms the vector of top features to tf-idf representation

# In[45]:


# Using a variable for the top k features to be selected
top_k_features=1000


text_processor = Pipeline([
    ('count vectorizer',CountVectorizer(stop_words=mystopwords,lowercase=True)),
    ('chi2score',SelectKBest(chi2,k=top_k_features)),
    ('tf_transformer',TfidfTransformer(use_idf=True))
])


# ### fit_transform Versus fit Versus Transform

# In[46]:


proc_text = text_processor.fit_transform(reviews.X,reviews.y)
proc_fit = text_processor.fit(reviews.X,reviews.y)


# In[47]:


# The tf-idf values for words in the first review that are among the top 1000 features is sparse matrix format
print(proc_text[0])


# Returning the original words that ended up in the final 1000 words for a particular comment can still be accomplished by the following two steps:
# + Find the index of the top 1000 features returned from the 'chi2score' transformation
# + Find the 'feature names', i.e. the words from the original text

# In[48]:


# proc_fit.named_steps['chi2score'].get_support(indices=True)


# In[49]:


proc_fit.named_steps['chi2score'].get_support(indices=True)[616]


# In[50]:


proc_fit.named_steps['count vectorizer'].get_feature_names()[23078]


# In[51]:


print(reviews.iloc[0,0])


# #### Old Pipeline 1

# In[ ]:


chi2(X_res,y_res)

k=1000

ch2_score = SelectKBest(chi2, k=k)

toxic_feature_tran = ch2_score.fit(X,y)

X_train_k = ch2_score.fit_transform(X, y)

X_test_k = ch2_score.transform(X_test)


# #### Old Pipeline 2

# In[22]:


count_vect = CountVectorizer(stop_words=mystopwords,lowercase=True)

X_train_counts = count_vect.fit_transform(X)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)

