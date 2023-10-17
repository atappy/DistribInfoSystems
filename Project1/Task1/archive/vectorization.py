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