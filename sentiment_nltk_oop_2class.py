from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # function for splitting data to train and test sets
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud, STOPWORDS
from subprocess import check_output
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re


class senti:

    def __init__(self, path):
        data = pd.read_csv(path)
        self.df = data[['text', 'sentiment']]  # , 'tweet_location'
        # print(type(self.df))
        # return df

    def split_data(self):
        self.train, self.test = train_test_split(self.df, random_state=0, test_size=0.1)
        return self.train, self.test

    def clean_data(self):
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
        return tweets

    def wordcloud_draw(self, title, data, color='black'):
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
        print(title)
        plt.figure(1, figsize=(13, 13))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

    # Extracting word features
    def get_words_in_tweets(self, tweets):
        all = []
        for (words, sentiment) in tweets:
            all.extend(words)
        return all

    def get_word_features(self, wordlist):
        wordlist = nltk.FreqDist(wordlist)
        self.features = wordlist.keys()
        return self.features

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def remove_neutral(self, data):
        data_pos = data[data['sentiment'] == 'Positive']
        data_pos = data_pos['text']
        data_neg = data[data['sentiment'] == 'Negative']
        data_neg = data_neg['text']
        return data_pos, data_neg

    def train_model(self, tweets):
        def extract_features(document):
            document_words = set(document)
            features = {}
            for word in self.features:
                features['contains(%s)' % word] = (word in document_words)
            return features
        training_set = nltk.classify.apply_features(extract_features, tweets)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return classifier

    def predict_label(self, classifier, data):
        print("PREDICTION")
        txt = []
        p_lbl = []
        a_lbl = []
        for i in range(len(data)):
            text = test.iloc[i]['text']
            txt.append(text)
            label = classifier.classify(obj.extract_features(text))
            p_lbl.append(label)
            a_lbl.append(data.iloc[i]["sentiment"])
            print(text, " ", label)
        result = {'Text': txt, 'Actual_sentiment': a_lbl, 'Predicted_sentiment': p_lbl}
        result = pd.DataFrame(result)
        return result

        # p_lbl = []
        # a_lbl = []
        # txt = []
        # for i in range(len(test)):
        #     obj = test.iloc[i]['text']
        #     txt.append(obj)
        #     label = classifier.classify(obj.extract_features(obj.split()))
        #     p_lbl.append(label)
        #     a_lbl.append(test.iloc[i]['sentiment'])

    def print_prediction(self, result):
        for i in range(len(result)):
            print(result.iloc[i])

    def acc_score(self, result):

        actual = result["Actual_sentiment"]
        predicted = result["Predicted_sentiment"]
        print("\nAccuracy: ", (accuracy_score(actual, predicted)) * 100)


if __name__ == "__main__":
    csv_path = "Dataset/Sentiment.csv"
    obj = senti(csv_path)
    train, test = obj.split_data()
    # train_pos, train_neg = obj.remove_neutral(train)
    # test_pos, test_neg = obj.remove_neutral(test)

    # visualize data
    # obj.wordcloud_draw('Positive Words',train_pos, 'white')
    # obj.wordcloud_draw('Negative words',train_neg)

    # cleaning data and removing stopwords
    tweets = obj.clean_data()
    w_features = obj.get_word_features(obj.get_words_in_tweets(tweets))

    # visualize features
    # obj.wordcloud_draw('Features',w_features)
    classifier = obj.train_model(tweets)
    print(test)
    result = obj.predict_label(classifier, test)
    obj.print_prediction(result)
    obj.acc_score(result)
