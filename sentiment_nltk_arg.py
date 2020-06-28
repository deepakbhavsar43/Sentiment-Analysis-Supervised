import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split  # function for splitting data to train and test sets
import nltk
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from subprocess import check_output

data = pd.read_csv('Dataset/Dataset.csv')
# Keeping only the neccessary columns
data = data[['text', 'sentiment']]

anssss = data['text']
print(anssss.head(10))

# Splitting the dataset into train and test set
train, test = train_test_split(data, test_size=0.1)
# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]
test = test[test.sentiment != "Neutral"]

train_pos = train[train['sentiment'] == 'Positive']
train_pos = train_pos['text']
train_neg = train[train['sentiment'] == 'Negative']
train_neg = train_neg['text']


def wordcloud_draw(data, color='black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                             if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT'
                             ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000
                          ).generate(cleaned_word)
    plt.figure(1, figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    print(cleaned_word)


# wordcloud_draw(data['text'])

tweets = []
stopwords_set = set(stopwords.words("english"))

for index, row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))

test_pos = test[test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg = test_neg['text']


# Extracting word features
def get_words_in_tweets(tweets):
    all = []
    for (words, sentiment) in tweets:
        all.extend(words)
    return all


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


w_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# wordcloud_draw(w_features)

# Training the Naive Bayes classifier
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

neg_cnt = 0
pos_cnt = 0
txt = []
p_lbl = []
a_lbl = []
for i in range(len(test)):
    obj = test.iloc[i]['text']
    txt.append(obj)
    label = classifier.classify(extract_features(obj.split()))
    p_lbl.append(label)
    a_lbl.append(test.iloc[i]['sentiment'])

result = {'Text': txt, 'Actual_sentiment': a_lbl, 'Predicted_sentiment': p_lbl}
result = pd.DataFrame(result)

for i in range(len(test)):
    print(result.iloc[i])

from sklearn.metrics import accuracy_score

actual = result["Actual_sentiment"]
predicted = result["Predicted_sentiment"]
print("\nAccuracy: ", (accuracy_score(actual, predicted)) * 100)
