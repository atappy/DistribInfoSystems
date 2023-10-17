import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
import pandas as pd
from scipy.sparse import lil_matrix
import pickle
from collections import defaultdict

def load_task1_data(F_reduced_dataset=False):
    dir_path = "Data/raw/"
    corpus = pd.read_json(f'{dir_path}corpus.jsonl', lines=True).set_index(['_id'])
    queries = pd.read_json(f'{dir_path}queries.jsonl', lines=True)[['_id', 'text']].set_index(['_id'])
    train_set = pd.read_table(f'{dir_path}task1_train.tsv')[["query-id","corpus-id"]].set_index(['query-id'])
    test_set = pd.read_table(f'{dir_path}task1_test.tsv')["query-id"]

    if F_reduced_dataset:
        corpus = corpus.head(15000)
        queries = queries.head(15000)
        train_set = train_set.head(5000)

    return corpus, queries, train_set, test_set

def remove_punctuation(text):
    return "".join([ch for ch in text if ch not in string.punctuation])

def tokenize(text, stemmer):
    tokens = word_tokenize(text)
    tokens = [remove_punctuation(token) for token in tokens]
    tokens = [stemmer.stem(word.lower()) for word in tokens if word not in stopwords.words('english') and len(word)>1]
    return tokens

def tokenize_corpus_queries(corpus, queries, stemmer):
    corpus['tokens'] = corpus['text'].apply(lambda x: tokenize(x, stemmer))
    queries['tokens'] = queries['text'].apply(lambda x: tokenize(x, stemmer))

def save_tokenized_corpus_queries(token_dir, corpus, queries):
    n_sample_per_file = 70000
    n_sample = len(corpus)
    n_files =  n_sample//n_sample_per_file + 1

    for i in range(0, n_files):
        first = i*n_sample_per_file
        last = min(first + n_sample_per_file, n_sample)
        corpus_tokens_part = corpus["tokens"].iloc[first:last]
        
        filename = f"{token_dir}corpus_tokens_{i:02d}.pkl"
        corpus_tokens_part.to_pickle(filename)
    
    queries["tokens"].to_pickle(f"{token_dir}queries_tokens.pkl")

def load_tokenized_corpus_queries(token_dir, corpus, queries):
    files_paths=listdir(token_dir)
    corpus_tokens_paths= [f"{token_dir+path}" for path in files_paths if 'corpus_tokens' in path and ".pkl" in path]
    corpus_tokens_paths.sort()
    dfs = [pd.read_pickle(path) for path in corpus_tokens_paths]
    corpus["tokens"] = pd.concat(dfs)
    queries["tokens"] = pd.read_pickle(f"{token_dir}queries_tokens.pkl")
    print("Files loaded")
    return corpus, queries

def count_terms(corpus, queries, vocab):
    C, Q, N = len(corpus), len(queries), len(vocab)
    count_corpus = lil_matrix((C,N), dtype=np.float32)
    count_queries = lil_matrix((Q,N), dtype=np.float32)
    vocab_index = dict(zip(vocab, range(len(vocab))))

    def defaultdict_value():
        return -1

    vocab_index = defaultdict(defaultdict_value, vocab_index)

    for i, c in enumerate(corpus['tokens'].to_list()):
        for term in c:
            index_term = vocab_index[term]
            count_corpus[i,index_term] += 1

    for i, q in enumerate(queries['tokens'].to_list()):
        for term in q:
            index_term = vocab_index[term]
            if index_term >= 0:
                count_queries[i,index_term] += 1
    return count_corpus, count_queries

def pca_tfidf(n_dim=500):
    with open("Data/tfidf/tfidf_corpus.npz","rb") as f:
        tfidf_corpus = pickle.load(f)
    with open("Data/tfidf/tfidf_queries.npz","rb") as f:
        tfidf_queries = pickle.load(f)
        # SVD
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=n_dim)
    svd.fit(tfidf_corpus)
    corpus_pca = svd.transform(tfidf_corpus)
    svd.explained_variance_ratio_.cumsum()[-1]

    return svd, corpus_pca

            
from collections import Counter, defaultdict
import math
import numpy as np

def compute_idf(documents):
    """
    Compute Inverse Document Frequency (IDF) for each term in all documents 
    
    documents : numpy array of tokens
    """
    
    total_documents = len(documents)
    word_document_count = defaultdict(int)

    average_number_words = 0 #global variable to compute average number of distinct words per document

    for document in documents:
        unique_words = set(document)
        average_number_words += len(unique_words)
        for word in unique_words:
            word_document_count[word] += 1
    average_number_words = average_number_words / total_documents

    idf = {}
    for word, count in word_document_count.items():
        idf[word] = math.log(total_documents / (count))

    idf = defaultdict(float,idf)

    return average_number_words, idf

def compute_tf(document, average_number_words, s = 0.2):
    """
    Compute Term Frequency (TF) for each term in a document and normalize it using the pivoted unique query normalization

    document : 
    average_number_words :
    s : normalization parameter

    """
    word_counts = Counter(document)
    unique_words_count = len(set(document)) # TODO : Use already computed unique_words_count in IDF ?
    tf = {word: (count / max(word_counts.values())) / ((1.0-s)*average_number_words + s*unique_words_count) for word, count in word_counts.items()}
    return tf

def vectorize(tokens, idf, average_number_words):
    """
    Compute TF-IDF weights for each term in all documents -> vectorize each document
    """
    vector = {}
    tf = compute_tf(tokens, average_number_words)
    vector.update({word: tf[word] * idf[word] for word in tf.keys()})
    return vector

def vectorize_corpus_queries(corpus, queries):
    corpus_tokens_array = corpus['tokens'].to_numpy()
    queries_tokens_array = queries['tokens'].to_numpy()
    average_number_words, idf = compute_idf(corpus_tokens_array)
    vectorized_corpus = [vectorize(document_tokens, idf, average_number_words) for document_tokens in corpus_tokens_array]
    vectorized_queries = [vectorize(query_tokens, idf, average_number_words) for query_tokens in queries_tokens_array]

    return vectorized_corpus, vectorized_queries


def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    v2 = defaultdict(float, v2)
    for i in v1:
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if sumxy == 0:
            result = 0
    else:
            result = sumxy/math.sqrt(sumxx*sumyy)
    return  result 

def k_search(query_vector, vectorized_corpus, corpus_ids, k=10):
    similarities = np.array([cosine_similarity(query_vector, doc_vec) for doc_vec in vectorized_corpus])
    corpus_ids = corpus_ids[similarities.argsort()[::-1]]

    return (corpus_ids[:k].to_list())


