# CMT316
A Random forrest classifier model for classifying BBC news articles into 5 separate categories.

In order to run the dataset needs to be in the same folder as the program datset found at: https://huggingface.co/datasets/SetFit/bbc-news

Required modules:
  -numpy
  -nltk
  -pandas
  -matplotlib
  -Wordcloud
  -sci-kit learn (sklearn)

The python file can be ran in the terminal using python Part_2.py. 

The program features:
  - Stopword filtering
  - Bag of words feature creation
  - n-gram feature creation
  - TF-IDF feature creation
  - Chi squared feature selection
  - The ability to find the optimal number of features and number of trees.

The default hyperparmaters for the model are 2000 features and a tree count of 500.

Optonal keyword arguments: (can be used together or seperatley)
  -f creates a development set to test for which is the optimal number of features to include in the model. The best   
     perfroming number of features is then automaticaly selected from the dataset for the final model.
  -t creates a development set and finds the accuracy for each of the number of trees sizes defined in the trees array . The best   
     perfroming number of trees is then automaticaly selected from the dataset for the final model.

Wordclouds for each category of news article will be saved in the directory it is ran in with the format <Category>_wordcloud.png
