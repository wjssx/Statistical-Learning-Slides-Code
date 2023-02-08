import gensim
from gensim import corpora
from pprint import pprint

# How to create a dictionary from a list of sentences?

documents = ["The Saudis are preparing a report that will acknowledge that",
             "Saudi journalist Jamal Khashoggi's death was the result of an",
             "interrogation that went wrong, one that was intended to lead",
             "to his abduction from Turkey, according to two sources."]

# Tokenize(split) the sentences into words
texts = [[text for text in doc.split()] for doc in documents]

# Create dictionary
dictionary = corpora.Dictionary(texts)

# Get information about the dictionary
print(dictionary)
print(dictionary.token2id)

documents_2 = ["The intersection graph of paths in trees",
               "Graph minors IV Widths of trees and well quasi ordering",
               "Graph minors A survey"]

texts_2 = [[text for text in doc.split()] for doc in documents_2]

dictionary.add_documents(texts_2)

print(dictionary.token2id)

new_corpus = [dictionary.doc2bow(text) for text in texts]

print(new_corpus)

from gensim import models

tfidf = models.TfidfModel(new_corpus)

corpus_tfidf = tfidf[new_corpus]
print(corpus_tfidf)

for i in range(len(corpus_tfidf)):
    print(corpus_tfidf[i])

string = 'the i first second name'
string_bow = dictionary.doc2bow(string.lower().split())
string_tfidf = tfidf[string_bow]
print(string_bow)
print(string_tfidf)