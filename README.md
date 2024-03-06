# CMT316
A Random forrest classifier model for classifying BBC news articles into 5 separate categories.

Required modules:
  -numpy
  -nltk
  -pandas
  -matplotlib
  -Wordcloud
  -sci-kit learn (sklearn)

The python file can be ran in the terminal using python Part_2.py. 

Optonal keyword arguments: (can be used together or seperatley)
  -f creates a development set to test for which is the optimal number of features to include in the model.
  -t creates a development set and finds the accuracy for each of the number of trees sizes defined in the trees array.

Wordclouds for each category of news article will be saved in the directory it is ran in with the format <Category>_wordcloud.png
