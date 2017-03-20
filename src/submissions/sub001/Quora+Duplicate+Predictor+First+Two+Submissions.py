
# coding: utf-8

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import resource
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# ------- Define modular methods for the task
def log_max_mem_usage():
    print(
        "Current all-time max memory: {} MB".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
        )
    )


# In[2]:

train_df = pd.read_csv('../input/train.csv')
train_df.dropna(inplace=True)  # For id: qid2 174364

test_df = pd.read_csv('../input/test.csv')

train_df.head(5)


# In[3]:

log_max_mem_usage()


# In[4]:

# %%time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# featurizers = [
#     ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3))),
#     ('word_tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 2)))
# ]

# char_weight = 0.4
# combined_featurizers = FeatureUnion(
#     featurizers,
#     n_jobs=7,
#     transformer_weights={
#         'char_tfidf': char_weight,
#         'word_tfidf': 1 - char_weight
#     }
# )

unique_questions = pd.Series(pd.concat([train_df.question1, train_df.question2]).unique())
# combined_featurizers.fit(unique_questions)

char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))  # featurizers[0][1]
word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))  # featurizers[1][1]
char_tfidf.fit(unique_questions)
word_tfidf.fit(unique_questions)

log_max_mem_usage()


# In[125]:

# %%time
samp = train_df.head(1000)
a = word_tfidf.transform(samp.question1)
b = word_tfidf.transform(samp.question2)


from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min, linear_kernel, cosine_similarity
# X, Y=None, metric='euclidean', n_jobs=1, **kwds)
# linear_kernel(a, b).diagonal()
# %timeit -n5 cosine_similarity(a, b, dense_output=False).diagonal()
get_ipython().magic(u'timeit -n5 np.dot(a, b.T).diagonal()')


# In[22]:

# %%time

def get_tfidf_features(data_df, batch=1000):
    i = 0

    word_dataset = np.array([])
    char_dataset = np.array([])

    while True:
        samp = data_df[i * batch: (i + 1) * batch]
        i += 1
        if i * batch % 10000 == 0:
            print(i * batch)

        if samp.empty:
            break

        word_res = np.dot(word_tfidf.transform(samp.question1), word_tfidf.transform(samp.question2).T).diagonal()
        char_res = np.dot(char_tfidf.transform(samp.question1), char_tfidf.transform(samp.question2).T).diagonal()

        word_dataset = np.concatenate([word_dataset, word_res])
        char_dataset = np.concatenate([char_dataset, char_res])

    return pd.DataFrame(dict(wv=word_dataset, cv=char_dataset), index=data_df.index)

log_max_mem_usage()


# In[6]:

from nltk.corpus import stopwords
from collections import Counter
import re


stops = set(stopwords.words("english"))
num_pattern = re.compile('[0-9]+')
math_pattern = re.compile('\[math\](.*)\[\/math\]')

nums = '01234567890'

def get_heuristic_scores(q1, q2):
#     n_q1 = {}
#     n_q2 = {}

#     for n in nums:
#         qc1 = q1.count(n)
#         qc2 = q2.count(n)
#         n_q1['q1_{}'.format(n)] = qc1
#         n_q2['q2_{}'.format(n)] = qc2

    exact_nums_q1 = num_pattern.findall(q1)
    exact_nums_q2 = num_pattern.findall(q2)
    
    math_q1 = math_pattern.findall(q1)
    math_q2 = math_pattern.findall(q2)

    num_exact_nums_match = len([n1 for n1 in exact_nums_q1 if n1 in exact_nums_q2])
    math_pattern_match = len([n1 for n1 in math_q1 if n1 in math_q2])
    
    is_q1_math = 1 * any(math_q1)
    is_q2_math = 1 * any(math_q2)
    is_both_math = is_q1_math * is_q2_math

#     qq2 = pd.Series(Counter([s for s in q1 if s.isupper()]))
#     qq1 = pd.Series(Counter([s for s in q2 if s.isupper()]))
    
#     sim_caps_rate = (qq1/qq2).mean()
#     num_caps_q1 = qq1.sum() 
#     num_caps_q2 = qq2.sum()

#     mean_caps_q1 = qq1.mean() 
#     mean_caps_q2 = qq2.mean()
    
    num_terms_q1 = len(q1.split())
    num_terms_q2 = len(q2.split())
    
    len_q1 = len(q1)
    len_q2 = len(q2)

    res = dict(
        num_exact_nums_match=num_exact_nums_match,
        math_pattern_match=math_pattern_match,
        is_q1_math=is_q1_math,
        is_q2_math=is_q2_math,
        is_both_math=is_both_math,
        length_diff=abs(len_q1 - len_q2),
        len_q1=len_q1,
        len_q2=len_q2,
        word_num_diff=abs(num_terms_q1 - num_terms_q2),
        num_terms_q1=num_terms_q1,
        num_terms_q2=num_terms_q2,
#         sim_caps_rate=sim_caps_rate,
#         mean_caps_q1=mean_caps_q1,
#         mean_caps_q2=mean_caps_q2,
#         num_caps_q1=num_caps_q1,
#         num_caps_q2=num_caps_q2,
    )
    
    # res.update(n_q1)
    # res.update(n_q2)
    
    return res


heuristics_feature_names = [
    'num_exact_nums_match',
    'math_pattern_match',
    'is_q1_math',
    'is_q2_math',
    'is_both_math',
    'length_diff',
    'len_q1',
    'len_q2',
    'word_num_diff',
    'num_terms_q1',
    'num_terms_q2',
]


def score_row(row):
    _, row = row


    q1 = row.question1
    q2 = row.question2

    if not all([set(q1.lower().split()).difference(stops), set(q2.lower().split()).difference(stops)]):
        print('here!')
        return {i: 0 for i in heuristics_feature_names}

    return get_heuristic_scores(q1, q2)


# In[ ]:




# In[7]:

# %%time
ds = []
samp = train_df[404000:]
for row in samp.iterrows():
    ds.append(score_row(row))

pd.DataFrame(ds, index=samp.index)


# In[8]:

# %%time
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import time

samp = train_df

def get_features(samp):
    start_time = time.time()
    
    tfidf_features = get_tfidf_features(samp)
    print('Finished computing tfidf features after {} seconds.'.format((time.time() - start_time)))

    heuristics_scores = []

    for row in samp.iterrows():
        if row[0] and row[0] % 10000 == 0:
            print(row[0])
            
        heuristics_scores.append(score_row(row))

    heuristics_scores = pd.DataFrame(heuristics_scores, index=samp.index)
    
    features = pd.concat([tfidf_features, heuristics_scores], axis=1)
    
    return features

X_train = get_features(samp)
y_train = samp.is_duplicate

log_max_mem_usage()


# In[9]:

# %%time

pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((1.0 * len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    print(scale)
    
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

X_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1029)

log_max_mem_usage()


# In[10]:

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, roc_auc_score, make_scorer


scaler = StandardScaler()

char_lm_model = LogisticRegression(C=100)
word_lm_model = LogisticRegression(C=1)
length_diff_lm_model = LogisticRegression(C=1)
word_num_diff_lm_model = LogisticRegression(C=1)

rf_model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, min_samples_split=3, n_jobs=-1)
lm_model = LogisticRegression()

def log_loss_scorer(model, X, y):
    return log_loss(y, model.predict_proba(X))

def fit_models(X, y):
    rf_model.fit(X_train, y_train)
    char_lm_model.fit(X_train.cv.values.reshape(-1, 1), y_train)
    word_lm_model.fit(X_train.wv.values.reshape(-1, 1), y_train)
    length_diff_lm_model.fit(X_train.length_diff.values.reshape(-1, 1), y_train)
    word_num_diff_lm_model.fit(X_train.word_num_diff.values.reshape(-1, 1), y_train)

def predict(X):
#     weights = dict(zip(X.columns, rf_model.feature_importances_))
    char_pred = char_lm_model.predict_proba(X.cv.values.reshape(-1, 1))[:, 1] # * weights['cv']
    word_pred = word_lm_model.predict_proba(X.wv.values.reshape(-1, 1))[:, 1] # * weights['wv']
    length_diff_pred = length_diff_lm_model.predict_proba(X.length_diff.values.reshape(-1, 1))[:, 1] # * weights['length_diff']
    word_num_diff_pred = word_num_diff_lm_model.predict_proba(X.word_num_diff.values.reshape(-1, 1))[:, 1] # * weights['word_num_diff']
    rf_pred = rf_model.predict_proba(X)[:, 1]

    return [char_pred, word_pred, length_diff_pred, word_num_diff_pred, rf_pred]


# In[11]:

# %%time

lm_model.fit(X_train, y_train)

print(log_loss_scorer(lm_model, X_valid, y_valid))


# In[12]:

# %%time

rf_model.fit(X_train, y_train)

print(log_loss_scorer(rf_model, X_valid, y_valid))


# In[13]:

# %%time
# fit_models(X_train, y_train)
# cross_val_score(rf_model, X_train, y_train, scoring=log_loss_scorer)


# In[14]:

print(roc_auc_score(y_valid, rf_model.predict_proba(X_valid)[:, 1]))


# In[15]:

# print(log_loss(y_test, np.mean(predict(X_test), axis=0)))
# print(roc_auc_score(y_test, np.mean(predict(X_test), axis=0)))
# print(roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))
# print(sum((y_test - (1 * (rf_model.predict_proba(X_test)[:, 1] < 0.5))) != 0))


# In[16]:

# for i, j in sorted(list(zip(X_test.columns, rf_model.feature_importances_)), key=lambda x: x[1], reverse=True):
#     print i, j


# In[17]:

# pd.concat([features, samp.is_duplicate], axis=1).groupby('is_duplicate').mean().T


# In[23]:

test_df.fillna('zxzxzx zxzxzx', inplace=True)


# In[26]:

1.0 * test_df.shape[0] / train_df.shape[0]


# In[27]:

# %%time
X_test = get_features(test_df)

log_max_mem_usage()


# In[34]:

# %%time
X_test.to_hdf('kaggle-quora', 'test-features-X_test')


# In[29]:

test_id = test_df.test_id

del(test_df)
del(train_df)


# In[30]:

import gc
gc.collect()


# In[32]:

# %%time
lm_sub = pd.DataFrame()

lm_sub['test_id'] = test_id
lm_sub['is_duplicate'] = lm_model.predict_proba(X_test)[:, 1]

lm_sub.to_csv('lm_submission.csv', index=False)
lm_sub.head()


# In[33]:

# %%time
rf_sub = pd.DataFrame()

rf_sub['test_id'] = test_id
rf_sub['is_duplicate'] = rf_model.predict_proba(X_test)[:, 1]

rf_sub.to_csv('rf_submission.csv', index=False)
rf_sub.head()


# In[ ]:

from IPython.display import FileLink
test_df = pd.read_csv('../input/test.csv')

scores_data = []
samp = train_df

for row in samp.iterrows():
    if row[0] % 5000 == 0:
        print(row[0])

    scores_data.append(score_row(row))

X = pd.DataFrame(scores_data)
is_duplicate_test = np.mean(predict(X), axis=0)

log_max_mem_usage()


# In[ ]:

del(train_df)

