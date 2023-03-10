# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python import keras

import string
import re  # regular expressions for text preprocessing
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from gensim.models import Word2Vec, KeyedVectors

from os.path import exists


def generator(frame, index_name):
    iterable = frame.iterrows()
    for i, doc in iterable:
        yield {
            "_index": index_name,
            "_source": doc.to_dict()
        }
        raise StopIteration


def userinput_query(client, searchreq):
    search_string = input("Search for: ")
    res = searchreq.query("query_string", query=search_string)
    userid = input("UserID: ")
    return res, int(userid)


def custom_metric(book, ratingsframe):
    book_code = book.isbn
    book_rating = ratingsframe.query("isbn == @book_code")  # dataframe with all the ratings for this book
    if book_rating.empty:
        mean_component = 0
    else:
        ratingslist = book_rating['rating'].to_list()
        mean_component = sum(ratingslist) / len(ratingslist)
        # get user rating
        try:
            user_component = book.user_rating
        except:
            print("kati pige strava")
        book.final_score = 0.5 * book.meta.score + 0.1 * mean_component + 0.4 * user_component
        return book.final_score


def display_custom_metric_results(results, df_ratings):
    sorted_results = sorted(results, key=lambda x: custom_metric(x, df_ratings), reverse=True)
    for i, hit in enumerate(sorted_results):
        print(f'Book n{i + 1} title: "{hit.book_title}" with score = {hit.final_score}"')


def get_training_data(userID):
    df_books = pd.read_csv("BX-Books.csv")
    df_rating = pd.read_csv("BX-Book-Ratings.csv")
    user_ratings = df_rating.query("uid == @userID")
    ratings = user_ratings['rating'].tolist()

    summaries = [df_books.query("isbn == @bookcode")['summary'].to_list() for bookcode in user_ratings.isbn.tolist()]
    categories = [df_books.query("isbn == @bookcode")['category'].to_list() for bookcode in user_ratings.isbn.tolist()]
    bookcodes = user_ratings.isbn.tolist()
    # kratame mono ta vivlia ta opoia exoun summaries giati proseksa oti iparxoun kai merika kena

    ratings_new = []
    for i, t in enumerate(summaries):
        if t:
            ratings_new.append(ratings[i])
    ratings_new = np.array(ratings_new)  # numpy array gia na perastei sto modelo
    summaries = [i[0] for i in summaries if i]  # as lists

    return [summaries, ratings_new, categories, bookcodes]


def create_elastic_client():
    df_books = pd.read_csv("BX-Books.csv")
    try:
        esclient = Elasticsearch("http://localhost:9200")
    except:
        print("Not connected to elasticsearch client")
    try:
        helpers.bulk(esclient, generator(df_books, index_name="books"))
    except:
        print("OK! bulking succesful")
    return esclient


def texts_preprocessing(texts):
    stop = set(stopwords.words("english"))
    if not texts:
        print("empty texts list given for preprocessing")
        return

    clean_texts = []
    for text in texts:
        # xrhsimopoioume regular expressions gia na afairesoume simvola apo tis lekseis
        text = text.lower()
        text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
        text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
        text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
        text = re.sub(
            f"[{re.escape(string.punctuation)}]", "", text
        )  # Remove punctuation

        text = text.split()
        text = [word for word in text if word not in stop or not word.isdigit()]
        text = [word for word in text if len(word) > 1]

        clean_texts.append(text)
    return clean_texts


def create_word2vec_model():
    df_books = pd.read_csv("BX-Books.csv")
    summaries = df_books['summary'].to_list()
    clean_summaries = texts_preprocessing(summaries)
    model = Word2Vec(clean_summaries, min_count=1, vector_size=100)
    model.save("wordtovecmodel.model")
    print("Word2Vec model created and saved in project directory")

    return model


def load_word_vectors():
    print('Word2vec embeddings loaded')
    return KeyedVectors.load("word2vec.wordvectors", mmap='r')


def get_document_vectors(list_of_texts, wv):
    clean_list = texts_preprocessing(list_of_texts)
    text_vectors = []
    for text in clean_list:
        word_vectors = []
        zero_vector = np.zeros(100)
        for word in text:
            try:
                word_vectors.append(wv[word])
            except KeyError:
                print("word not found")
                word_vectors.append(zero_vector)
        if word_vectors:
            word_vectors = np.asarray(word_vectors)
            text_vector = word_vectors.mean(axis=0)  # text vector is the average of its word vectors
            text_vector = text_vector / np.linalg.norm(text_vector)
        else:
            text_vector = zero_vector
        text_vectors.append(np.asarray(text_vector))

    return np.asarray(text_vectors)


def get_sequential_model(training_data_x, training_data_y, embeddings):
    training_vectors = get_document_vectors(training_data_x, embeddings)
    # create model

    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=training_vectors[0].shape))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    print("Training Neural Network . . .")
    history = model.fit(training_vectors, training_data_y, epochs=500, verbose=0)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch  # to use hist.tail method
    print("Model trained")
    return model, hist


def update_ratings_with_sequential(seq, query_results, user_rated_codes, wv):
    for book in query_results:
        if book.isbn not in user_rated_codes:  # gia ta vivlia pou den ehoun vathmologithei apton xristi
            summary_vector = get_document_vectors([book.summary], wv)
            book.user_rating = seq.predict(summary_vector)[0][0]
            if book.user_rating > 10:
                book.user_rating = 10.0
            if book.user_rating < 0:
                book.user_rating = 0.0
    return query_results


def display_elasticsearch_results(res):
    for i, hit in enumerate(res):
        print(f'Book n{i + 1} title: "{hit.book_title}" with elasticscore = {hit.meta.score}"')


if __name__ == '__main__':

    if not exists("BooksClustered.csv"):
        df_books = pd.read_csv("BX-Books.csv")
        df_ratings = pd.read_csv("BX-Book-Ratings.csv")
        if not exists("wordtovecmodel.model"):
            model = create_word2vec_model()
            if not exists("word2vec.wordvectors"):
                model.wv.save("word2vec.wordvectors")

        wordvectors = load_word_vectors()

        summaries_list = df_books['summary'].tolist()
        print("Transforming summaries to vectors . . .")
        vectors_of_summaries = get_document_vectors(summaries_list, wordvectors)
        # nea methodos get_document_vectors kanei normalize ta text vectors gia na ehoume sxesi EuclDist-Cos Similarity
        print("Done")
        p = PCA(2)
        print("Projecting to 2D . . .")
        vectors_2d = p.fit_transform(vectors_of_summaries)
        print("Done")
        print("Applying Kmeans algorithm . . . ")
        km = KMeans(n_clusters=50).fit(vectors_of_summaries)
        print("Done")

        labels = km.labels_

        df_books['cluster'] = labels
        df_books.to_csv("BooksClustered.csv")
        plt.figure()
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=labels.astype(float), s=1)
        plt.show()

    df_books = pd.read_csv("BooksClustered.csv")
    df_users = pd.read_csv("BX-Users.csv")
    df_ratings = pd.read_csv("BX-Book-Ratings.csv")

    for n in range(0, 50):
        df_cluster = df_books.loc[df_books['cluster'] == n]

        df_cluster_ratings = df_ratings.loc[df_ratings['isbn'].isin(df_cluster['isbn'].tolist())]
        df_cluster_users = df_users.loc[df_users['uid'].isin(df_cluster_ratings['uid'].tolist())]

        most_frequent_category = df_cluster['category'].mode()
        print(f'Top genre in cluster {n} is {most_frequent_category.tolist()[0]}')
        avg_rating = df_cluster_ratings["rating"].mean()
        print(f'The average ratings were {avg_rating}')
        avg_age = df_cluster_users['age'].mean()
        print(f'The average age of the people who rated is {avg_age} ')
        most_frequent_location = df_cluster_users['location'].mode().tolist()[0]
        print(f'Most of them were in {most_frequent_location}')
        print("")

    cluster_num = input("Choose Cluster to plot detailed histograms OR press 0 to close program:\n")
    while cluster_num != '0':
        cluster = df_books.loc[df_books['cluster'] == int(cluster_num)]
        cluster_ratings = df_ratings.loc[df_ratings['isbn'].isin(df_cluster['isbn'].tolist())]
        cluster_users = df_users.loc[df_users['uid'].isin(df_cluster_ratings['uid'].tolist())]

        plt.figure(f'Cluster {cluster_num}')
        filtered_by_age_val_counts = cluster_users.groupby('age').filter(lambda x: len(x) > 5)
        filtered_by_age_val_counts['age'].value_counts().sort_index().plot(kind='bar', title='Age of Users who rated')
        plt.show()

        plt.figure(f'Cluster {cluster_num}')
        cluster_ratings['rating'].value_counts().sort_index().plot(kind='bar', title='Ratings')
        plt.show()

        cluster_users['location']=cluster_users['location'].apply(lambda x: x.split()[-1])
        filtered_by_country_val_counts = cluster_users.groupby('location').filter(lambda x: len(x) > 5)
        plt.figure(f'Cluster {cluster_num}')
        filtered_by_country_val_counts['location'].value_counts().plot(kind='bar',title='Countries')
        plt.show()

        cluster_num = input("Choose Cluster to plot detailed histograms OR press 0 to exit program:\n")

