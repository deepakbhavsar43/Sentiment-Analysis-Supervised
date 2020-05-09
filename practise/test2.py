import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
import nltk.corpus as corpus


# stopWords = corpus.stopwords.words('english')
# vectorizer = CountVectorizer(stop_words = stopWords)

data = pd.read_csv('Dataset/Sentiment.csv')

# Keeping only the neccessary columns
df = data[['text', 'sentiment']]

# print(df.head(10))

lem = nltk.WordNetLemmatizer()
pstem = nltk.PorterStemmer()

# print(df.loc[0]['text'])
# exit()

stop_words = corpus.stopwords.words('english')
for i in range(len(df.index)):
    text = df.loc[i]['text']
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
for j in range(len(tokens)):
    tokens[j] = lem.lemmatize(tokens[j])
    tokens[j] = pstem.stem(tokens[j])
tokens_sent = ' '.join(tokens)
df.at[i, "text"] = tokens_sent

print(df.head())