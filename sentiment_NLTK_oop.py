from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from userInput import *
import pandas as pd
import pickle
import nltk

class sentiment_analysis:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        # Keeping only the neccessary columns
        self.data = self.data[['text', 'sentiment']]

    def split_data(self):
        # Splitting the dataset into train and test set
        self.train, self.test = train_test_split(self.data, test_size=0.1)
        return self.train, self.test

    def wordcloud_draw(self, data, color='black'):
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

    #         print(cleaned_word)

    def separate(self, data):
        positive = data[data['sentiment'] == 'Positive']
        positive = positive['text']
        neutral = data[data['sentiment'] == 'Neutral']
        neutral = neutral['text']
        negative = data[data['sentiment'] == 'Negative']
        negative = negative['text']
        return positive, neutral, negative

    def clean_data(self, data):
        tweets = []
        stopwords_set = set(stopwords.words("english"))

        for index, row in data.iterrows():
            words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
            words_cleaned = [word for word in words_filtered
                             if 'http' not in word
                             and not word.startswith('@')
                             and not word.startswith('#')
                             and word != 'RT']
            words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
            tweets.append((words_without_stopwords, row.sentiment))
        return tweets

    # Extracting word features
    def get_word_features(self, tweets):
        all = []
        for (words, sentiment) in tweets:
            all.extend(words)

        wordlist = nltk.FreqDist(all)
        features = wordlist.keys()
        return features

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in w_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def train_model(self, tweets):
        # Training the Naive Bayes classifier
        training_set = nltk.classify.apply_features(sentiment.extract_features, tweets)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return classifier

    def wr_pickle(self, train, model):
        # creating pickle file
        outfile = open(model, 'wb'), tweets
        # saving trained model in pickle file
        pickle.dump(train, outfile)
        outfile.close()

    def rd_pickle(self, model):
        # opening pickle file
        infile = open(model, 'rb')
        # loading the saved model into variable newTraining
        self.newTraining = pickle.load(infile)
        return self.newTraining

    def predict(self, classifier, data):
        neg_cnt = 0
        pos_cnt = 0
        txt = []
        p_lbl = []
        a_lbl = []
        for i in range(len(data)):
            obj = data.iloc[i]['text']
            txt.append(obj)
            label = classifier.classify(sentiment.extract_features(obj.split()))
            p_lbl.append(label)
            a_lbl.append(test.iloc[i]['sentiment'])

        self.result = {'Text': txt, 'Actual_sentiment': a_lbl, 'Predicted_sentiment': p_lbl}
        self.result = pd.DataFrame(self.result)
        return self.result

    def wr_pickle(self, train, model):
        # creating pickle file
        outfile = open(model, 'wb')
        # saving trained model in pickle file
        pickle.dump(train, outfile)
        outfile.close()

    def rd_pickle(self, model):
        # opening pickle file
        infile = open(model, 'rb')
        # loading the saved model into variable newTraining
        self.newTraining = pickle.load(infile)
        return self.newTraining

    def acc_score(self):
        actual = self.result["Actual_sentiment"]
        predicted = self.result["Predicted_sentiment"]
        print("\nAccuracy of Sentiment Analysis : ", (accuracy_score(actual, predicted)) * 100)

    def total_sentiment(self):
        actual_number = {}
        for i in ['Positive', 'Negative', "Neutral"]:
            act = self.result[self.result['Actual_sentiment'] == i]
            total = len(act)
            actual_number[i] = total

        predicted_number = {}
        for i in ['Positive', 'Negative', "Neutral"]:
            pred = self.result[self.result['Predicted_sentiment'] == i]
            total = len(pred)
            predicted_number[i] = total

        print("Actual :\n",actual_number,"\n","Predicted:\n",predicted_number)



if __name__ == "__main__":
    CSVFileName = "Dataset/sentiment.csv"
    Trained_Model_File = "Trained_Model/trained_data"
    sentiment = sentiment_analysis(CSVFileName)
    train, test = sentiment.split_data()
    train_pos, train_neut, train_neg = sentiment.separate(train)
    #     sentiment.wordcloud_draw(train_pos)
    tweets = sentiment.clean_data(train)
    w_features = sentiment.get_word_features(tweets)
    if args.train:
        classifier = sentiment.train_model(tweets)
        sentiment.wr_pickle(classifier, Trained_Model_File)
        print("Model Trained...")
    elif args.test:
        classifier = sentiment.rd_pickle(Trained_Model_File)
        prediction = sentiment.predict(classifier, test)
        print(prediction)
        sentiment.acc_score()
        sentiment.total_sentiment()