import  pandas as pd
corpus = ['this is the first document',
          'this document is the second document',
          'and this is the third one',
          'is this the first document']

def display_features(features,feature_names):
    df = pd.DataFrame(data=features, columns = feature_names)
    print(df)




from sklearn.feature_extraction.text import CountVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features



bow_vectorizer, bow_features = bow_extractor(corpus)
print(bow_features.todense())

feature_names = bow_vectorizer.get_feature_names()

print(feature_names)

features = bow_features.todense()
display_features(features, feature_names)


from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix



import numpy as np

feature_names = bow_vectorizer.get_feature_names()
tfidf_trans, tdidf_features = tfidf_transformer(bow_features)

features = np.round(tdidf_features.todense(), 2)
display_features(features, feature_names)

