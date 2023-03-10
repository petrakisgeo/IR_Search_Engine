# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search

import pandas as pd
import numpy as np
from tensorflow.python import keras

import string
import re  # regular expressions for text preprocessing
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.text import Tokenizer
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
    summaries = [i[0] for i in summaries if i]  # list of strings (not list of lists)

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


def update_ratings_with_sequential(seq, query_results, userid, ratings, wv):
    user_rated_codes=ratings.query("uid == @userid").isbn.tolist()
    for book in query_results:
        if book.isbn in user_rated_codes:  # gia ta vivlia pou den ehoun vathmologithei apton xristi
            rating = ratings.query("isbn== @book.isbn and userid == @uid")
            book.user_rating = rating.iloc[0]['rating']
        else:
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

    df_ratings = pd.read_csv("BX-Book-Ratings.csv")
    if not exists("wordtovecmodel.model"):
        model = create_word2vec_model()
        if not exists("word2vec.wordvectors"):
            model.wv.save("word2vec.wordvectors")

    wordvectors = load_word_vectors()

    esclient = create_elastic_client()
    sm = Search(using=esclient)  # elasticsearch DSL gia pio katanohta queries
    query_res, userid = userinput_query(esclient, sm)  # userinput

    summaries_training, ratings, categories_of_rated, bookcodes = get_training_data(
        userid)  # oi perilipseis pou ehoun kritikes apton xrhsth kai oi kritikes
    print(ratings)
    model, history = get_sequential_model(summaries_training, ratings, wordvectors)


    updated_results = update_ratings_with_sequential(model, query_res, userid, df_ratings,  wordvectors)
    display_elasticsearch_results(query_res)
    print("After rating predictions:")
    display_custom_metric_results(updated_results, df_ratings)


