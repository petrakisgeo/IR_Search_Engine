# IR_Search_Engine
An information retrieval system for the "Book Crossing Dataset" (Freiburg Information Institute) in an Anaconda python environment using Elasticsearch python API and the Pandas module.

Version 1: Returns the results of a search query based on Elasticsearch score and user reviews

Version 2: Using Word Embeddings (Google's "Word2Vec" model) and ML (Tensorflow and Keras modules) we predict the reviews of users for their unreviewed books based on the book summaries by training the model on the user's previously reviewed books.

Version 3: The program no longer returns search results. Using the scikit-learn python module the books are clustered using the Kmeans algorithm and Partial Component Analysis on their Embeddings and then statistics are extracted for the clusters

How to use:
  1. Install and run the Elasticsearch Client
  2. Download the dataset through http://www2.informatik.uni-freiburg.de/~cziegler/BX/ in .csv format and extract the package. The .csv files need to be in the same directory with the .py file for the program to work
  3. Further instructions provided on execution of the program
