import gensim
from gensim.models import Word2Vec

# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
             ['this', 'is', 'the', 'second', 'sentence'],
             ['yet', 'another', 'sentence'],
             ['one', 'more', 'sentence'],
             ['and', 'the', 'final', 'sentence']]

# train model
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
# model.build_vocab(sentences, update=True)
# model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)

print(model)
vector = list(model.wv['first'])
print(vector)

sims = model.wv.most_similar('first', topn=10)  # get other similar words
print(sims)

word_vectors = model.wv
print(word_vectors)

for index, word in enumerate(model.wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(model.wv.index_to_key)} is {word}")

similarity = word_vectors.similarity('first', 'second')
print(similarity)

result = word_vectors.similar_by_word("first")
print(result)

_idx = model.wv.key_to_index["first"]
print(_idx)
