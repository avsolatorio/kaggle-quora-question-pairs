# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pylab as plt

import resource
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
# ------- Define modular methods for the task
def log_max_mem_usage():
    print(
        "Current all-time max memory: {} MB".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000
        )
    )


# Features and EDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from fuzzywuzzy import fuzz
import fuzzy
import ngram
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from collections import Counter
import re

# Parallelization
import multiprocessing as mp
from joblib import Parallel, delayed

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    log_loss, roc_auc_score, make_scorer, f1_score, recall_score, precision_score,
    normalized_mutual_info_score, mutual_info_score
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
import fasttext
import xgboost as xgb
from gensim.models import Word2Vec

# Utils
import psutil
from datetime import datetime
import time
import os
import cPickle


def auto_reshape_cos(f):
    def _cos_sim(x, y):
        if isinstance(x, list):
            x = np.array(x)

        if isinstance(y, list):
            y = np.array(y)

        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        if len(y.shape) == 1:
            y = y.reshape(1, -1)
            
        return f(x, y)
    return _cos_sim


cosine_similarity = auto_reshape_cos(cosine_similarity)

def einsum_dot(a, b):
    return np.einsum('ij,ji->i', a, b.T)


def l2_norm(x):
    return np.linalg.norm(x, ord=2, axis=1)


def fast_pairwise_cos_sim(a, b):
    an = l2_norm(a)
    bn = l2_norm(b)
    ap = (a / an.reshape(a.shape[0], 1)).reshape(np.prod(a.shape))
    bp = (b / bn.reshape(b.shape[0], 1)).reshape(np.prod(b.shape))
    cp = ap * bp
    c = cp.reshape(a.shape).sum(axis=1)

    return c


def einsum_pairwise_cos_sim(a, b):
    an = l2_norm(a)
    bn = l2_norm(b)
    ap = (a / an.reshape(a.shape[0], 1))
    bp = (b / bn.reshape(b.shape[0], 1))

    return einsum_dot(ap, bp)


def np_pairwise_cos_sim(a, b):
    an = l2_norm(a)
    bn = l2_norm(b)
    a = (a / an.reshape(a.shape[0], 1))
    b = (b / bn.reshape(b.shape[0], 1))
    c = a.dot(b.T).diagonal()

    return c

# Distances
# https://brenocon.com/blog/2012/03/cosine-similarity-pearson-correlation-and-ols-coefficients/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
def rowwise_minkowski_distance(a, b, p):
    return np.linalg.norm((a-b), ord=p, axis=1)


def rowwise_correlation(a, b):
    a = (a - a.mean(axis=1).reshape(a.shape[0], 1))
    b = (b - b.mean(axis=1).reshape(b.shape[0], 1))
    return einsum_pairwise_cos_sim(a, b)


def ols_coef(a, b, norm_var):
    # one-variable linear regression coefficient.
    # norm_var determines which vector to use for normalization.
    c = einsum_dot(a, b)

    if norm_var == 0:
        n = l2_norm(a)
    elif norm_var == 1:
        n = l2_norm(b)
    else:
        raise ValueError('norm_var value unknown.')

    return c / (n**2)


def ols_coef_with_intercept(a, b, norm_var):
    # one-variable linear regression coefficient.
    # norm_var determines which vector to use for normalization.
    if norm_var == 0:
        a = (a - a.mean(axis=1).reshape(a.shape[0], 1))
    elif norm_var == 1:
        b = (b - b.mean(axis=1).reshape(b.shape[0], 1))
    else:
        raise ValueError('norm_var value unknown.')

    return ols_coef(a, b, norm_var=norm_var)


def to_binary(x):
    return 1.0 * (x > 0)


def get_C(a, b, returns='all'):
    not_a = 1 - a
    not_b = 1 - b

    if returns == 'all':
        ntt = (a * b).sum(axis=1)
        ntf = (a * not_b).sum(axis=1)
        nff = (not_a * not_b).sum(axis=1)
        nft = (not_a * b).sum(axis=1)
        return (ntt, ntf, nff, nft)
    elif returns == 'mix':
        ntf = (a * not_b).sum(axis=1)
        nft = (not_a * b).sum(axis=1)
        return (ntf, nft)
    else:
        raise ValueError('Unknown returns parameter!')


def rowwise_binary_hamming_distance(a, b, binary_input):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    return np.logical_xor(a, b).sum(axis=1) / float(a.shape[1])


def rowwise_binary_jaccard_similarity(a, b, binary_input):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    return 1.0 * np.logical_and(a, b).sum(axis=1) / np.logical_or(a, b).sum(axis=1)


def rowwise_chebyshev_distance(a, b):
    return np.abs(a - b).max(axis=1)


def rowwise_canberra_distance(a, b):
    return (np.abs(a - b) / (np.abs(a) + np.abs(b))).sum(axis=1)


def rowwise_braycurtis_distance(a, b):
    return np.abs(a - b).sum(axis=1) / np.abs(a + b).sum(axis=1)


def rowwise_yule_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT, cTF, cFF, cFT = get_C(a, b)
    else:
        cTT = pre_computed_vars.get('cTT')
        cTF = pre_computed_vars.get('cTF')
        cFF = pre_computed_vars.get('cFF')
        cFT = pre_computed_vars.get('cFT')

    R = 2.0 * cTF * cFT

    return R / ((cTT * cFF) + (R / 2.0))


def rowwise_dice_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT, cTF, cFF, cFT = get_C(a, b)
    else:
        cTT = pre_computed_vars.get('cTT')
        cTF = pre_computed_vars.get('cTF')
        cFT = pre_computed_vars.get('cFT')

    return (cTF + cFT) / ((2.* cTT) + cFT + cTF)


def rowwise_kulsinski_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT, cTF, cFF, cFT = get_C(a, b)
    else:
        cTT = pre_computed_vars.get('cTT')
        cTF = pre_computed_vars.get('cTF')
        cFT = pre_computed_vars.get('cFT')

    n = float(a.shape[1])

    return (cTF + cFT - cTT + n) / (cFT + cTF + n)


def rowwise_rogerstanimoto_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT, cTF, cFF, cFT = get_C(a, b)
    else:
        cTT = pre_computed_vars.get('cTT')
        cTF = pre_computed_vars.get('cTF')
        cFF = pre_computed_vars.get('cFF')
        cFT = pre_computed_vars.get('cFT')

    R = 2.0 * (cTF + cFT)

    return R / (cTT + cFF + R)


def rowwise_russellrao_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT = (a * b).sum(axis=1)
    else:
        cTT = pre_computed_vars.get('cTT')

    n = float(a.shape[1])

    return (n - cTT) / n


def rowwise_sokalmichener_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT, cTF, cFF, cFT = get_C(a, b)
    else:
        cTT = pre_computed_vars.get('cTT')
        cTF = pre_computed_vars.get('cTF')
        cFF = pre_computed_vars.get('cFF')
        cFT = pre_computed_vars.get('cFT')

    R = 2.0 * (cTF + cFT)
    S = cFF + cTT

    return R / (S + R)


def rowwise_sokalsneath_distance(a, b, binary_input, pre_computed_vars={}):
    if not binary_input:
        # Input is assumed to be a vector ranging from -inf to inf
        # Convert to binary
        a = to_binary(a)
        b = to_binary(b)

    if not pre_computed_vars:
        cTT, cTF, cFF, cFT = get_C(a, b)
    else:
        cTT = pre_computed_vars.get('cTT')
        cTF = pre_computed_vars.get('cTF')
        cFF = pre_computed_vars.get('cFF')
        cFT = pre_computed_vars.get('cFT')

    R = 2.0 * (cTF + cFT)

    return R / (cTT + R)


def rowwise_tanimoto_similarity(a, b):
    num = einsum_dot(a, b)
    an_sq = l2_norm(a)**2.0
    bn_sq = l2_norm(b)**2.0

    return num / (an_sq + bn_sq - num)


def load_stopwords(extra_whitelist=[]):
    whitelist = set(['not'])
    whitelist.update(extra_whitelist)
    
    stops = set(stopwords.words("english"))
    stops.difference_update(whitelist)

    return stops


def print_question_pairs(df, sample=True, n=10):
    if sample:
        df = df.sample(n=n)

    for row in df.iterrows():
        _, row = row
        print row.question1
        print row.question2
        try:
            print row.is_duplicate
        except:
            pass
        print


def log_loss_scorer(model, X, y):
    return log_loss(y, model.predict_proba(X))


def train_xgb(
    x_train, x_valid, y_train, y_valid,
    params={}, num_boost_round=1000,
    early_stopping_rounds=100, verbose_eval=10,
    max_depth=7
):
    if not params:
        # Set our parameters for xgboost
        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
        params['eta'] = 0.02
        params['max_depth'] = max_depth

    d_train = xgb.DMatrix(x_train, label=y_train)
    watchlist = [(d_train, 'train')]
    
    if x_valid is not None:
        d_valid = xgb.DMatrix(x_valid, label=y_valid)
        watchlist.append((d_valid, 'valid'))

    bst = xgb.train(
        params, d_train, num_boost_round=num_boost_round,
        evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval
    )
    
    return bst


def load_google_news_w2v(path='GoogleNews-vectors-negative300.bin.gz'):
    wvmodel = Word2Vec.load_word2vec_format(path, binary=True)
    return wvmodel


def train_word2vec(
    tokenized_corpus,
    pre_trained_model='GoogleNews-vectors-negative300.bin.gz',
    size=300,
    iter=100,
    min_count=1,
    negative=10,
    workers=7,
    min_alpha=0.0001,
    window=5,
    binary=True,
):
    # https://github.com/RaRe-Technologies/gensim/issues/1245
    # List of tokenized questions.
    # e.g. ['What', 'is', 'the', 'step', 'by', 'step', 'guide', 'to', 'invest', 'in', 'share', 'market', 'in', 'india']
    # pre_trained_model can be any pre trained model that gensim accepts, e.g., Glove or GoogleNews word2vec

    # Initialize model
    word_vectors = Word2Vec(
        size=size, iter=iter, min_count=min_count, negative=negative, workers=workers,
        min_alpha=min_alpha, window=window,
    )

    # Initialize vocab
    word_vectors.build_vocab(tokenized_corpus)

    # Initialize vectors in local model with with vectors from pre-trained model with overlapping vocabulary.
    # Set `lockf` to 1 for re-training
    word_vectors.intersect_word2vec_format(pre_trained_model, lockf=1, binary=binary)

    # Adjust pre-trained vectors to adapt its distribution with that of the local data via retraining.
    word_vectors.train(tokenized_corpus)

    return word_vectors
