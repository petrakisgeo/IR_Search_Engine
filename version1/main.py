# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search
import pandas as pd


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
    return res


def custom_metric(book,userid, ratingsframe):
    book_code = book.isbn
    book_rating = ratingsframe.query("isbn == @book_code")  # dataframe with all the ratings for this book
    if book_rating.empty:
        print("no ratings for this book")
        book.custom_score=0.5*book.meta.score
        return book.custom_score
    else:
        ratingslist = book_rating['rating'].to_list()
        mean_component = sum(ratingslist) / len(ratingslist)
        # get user rating
        user_rating_frame = ratingsframe.query("isbn == @book_code and uid == @userid") #dataframe with empty
        if user_rating_frame.empty:
            user_component=0
        else:
            # exoume apomonwsei tin grammi me to rating apton xristi kai prepei na eksagoume tin vathmologia apto dataframe
            user_component=user_rating_frame.iloc[0]['rating']
        book.custom_score= 0.5*book.meta.score + 0.4*user_component + 0.1*mean_component
        return book.custom_score

def display_custom_metric_results(results,df_ratings):
    userID=input("For results based on custom metric, enter userID: ")
    userID=int(userID)
    sorted_results = sorted(results, key=lambda x: custom_metric(x, userID, df_ratings), reverse=True)
    for i, hit in enumerate(sorted_results):
        print(f'Book n{i+1} title: "{hit.book_title}" with custom metric score: {hit.custom_score}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_books = pd.read_csv("BX-Books.csv")
    df_rating = pd.read_csv("BX-Book-Ratings.csv")
    try:
        esclient = Elasticsearch("http://localhost:9200")
        esclient.info()
    except:
        print("Not connected to elasticsearch client")
    try:
        helpers.bulk(esclient, generator(df_books, index_name="books"))
    except:
        print("OK! bulking succesful")
    # after connecting to client and feeding the data, we listen for user input
    sm = Search(using=esclient)
    query_res = userinput_query(esclient, sm)

    display_custom_metric_results(query_res,df_rating)
