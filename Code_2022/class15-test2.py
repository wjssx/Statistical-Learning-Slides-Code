from gensim import corpora, models, similarities
from pprint import pprint
import warnings

f = open('data/LDA_test.txt')
stop_list = set('for a of the and to in'.split())

texts = [[
    word for word in line.strip().lower().split() if word not in stop_list
] for line in f]
print('Text = ')
pprint(texts)

dictionary = corpora.Dictionary(texts)
print(dictionary)

V = len(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]
corpus_tfidf = models.TfidfModel(corpus)[corpus]
corpus_tfidf = corpus

print('\nTF-IDF:')
for c in corpus_tfidf:
    print(c)

print('\nLSI Model:')
lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)
topic_result = [a for a in lsi[corpus_tfidf]]
pprint(topic_result)

print('\nLSI Topics:')
pprint(lsi.print_topics(num_topics=2, num_words=5))

print('\nLDA Model:')
num_topics = 2
lda = models.LdaModel(
    corpus_tfidf,
    num_topics=num_topics,
    id2word=dictionary,
    alpha='auto',
    eta='auto',
    minimum_probability=0.001,
    passes=10)
doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
print('Document-Topic:')
pprint(doc_topic)
