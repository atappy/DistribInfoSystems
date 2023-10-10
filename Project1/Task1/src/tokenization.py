import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
import pandas as pd

def load_task1_data(F_reduced_dataset=False):
    """
    Load data useful for task1 as dataframes
    """

    dir_path = "Data/raw/"

    corpus = pd.read_json(f'{dir_path}corpus.jsonl', lines=True).set_index(['_id'])
    queries = pd.read_json(f'{dir_path}queries.jsonl', lines=True)[['_id', 'text']].set_index(['_id'])
    train_set = pd.read_table(f'{dir_path}task1_train.tsv')[["query-id","corpus-id"]].set_index(['query-id'])

    if F_reduced_dataset:
        corpus = corpus.head(15000)
        queries = queries.head(15000)
        train_set = train_set.head(5000)

    return corpus, queries, train_set

def remove_punctuation(text):
    """
    Remove any punctuation charachter from a text (str)
    """
    return "".join([ch for ch in text if ch not in string.punctuation])

def tokenize(text, stemmer):
    """
    Transform a text (str) in a list of stemmed words (list[str])
    """

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
    print("Files saved")

def load_tokenized_corpus_queries(token_dir, corpus, queries):
    files_paths=listdir(token_dir)
    corpus_tokens_paths= [f"{token_dir+path}" for path in files_paths if 'corpus_tokens' in path and ".pkl" in path]
    corpus_tokens_paths.sort()
    dfs = [pd.read_pickle(path) for path in corpus_tokens_paths]
    corpus["tokens"] = pd.concat(dfs)
    queries["tokens"] = pd.read_pickle(f"{token_dir}queries_tokens.pkl")
    print("Files loaded")
    return corpus, queries

def generate_vocab_from_corpus(corpus):
    # Own method
    tokens = corpus['tokens']
    vocab_custom = pd.DataFrame(sorted(list(set([ x for y in tokens for x in y]))), columns=["word"])
    
    # TFID method
    texts = corpus['text']
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df = 1, stop_words = 'english')
    tf.fit_transform(texts)
    vocab_TFID = pd.DataFrame(sorted(list(tf.vocabulary_.keys())), columns=["word"])

    # Vocab with word and method 
    vocab = vocab_custom.merge(vocab_TFID, on=['word'], how='outer', indicator=True)
    vocab.columns = ["word", "method"]
    vocab.replace({
        "both":"Both",
        "left_only":"Custom",
        "right_only":"TFID"
    }, inplace=True)


    return vocab

def save_vocab(vocab:pd.DataFrame, vocab_path):
    vocab.to_csv(vocab_path, index=False)
