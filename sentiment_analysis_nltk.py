from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# from userInput import *
import streamlit as st
import pandas as pd
import pickle
import nltk


class sentiment_analysis:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        # Keeping only the neccessary columns
        self.data = self.data[['text', 'sentiment']]
        # self.data = sentiment_analysis.choose_ratio(self.data)

    def choose_ratio(self):
        tweet = []
        sentiment = []
        for s in ['Negative', 'Neutral', 'Positive']:
            temp = self.data[self.data['sentiment'] == s]
            for i in range(0, 300):
                txt = temp.iloc[i]['text']
                sent = temp.iloc[i]['sentiment']
                tweet.append(txt)
                sentiment.append(sent)
        self.data = pd.DataFrame({
            "text": tweet,
            "sentiment": sentiment
        })
        return self.data

    def split_data(self):
        # Splitting the dataset into train and test set
        self.train, self.test = train_test_split(self.data, test_size=0.1)
        return self.data, self.train, self.test

    # def wordcloud_draw(self, data, color='black'):
    #     words = ' '.join(data)
    #     cleaned_word = " ".join([word for word in words.split()
    #                              if 'http' not in word
    #                              and not word.startswith('@')
    #                              and not word.startswith('#')
    #                              and word != 'RT'
    #                              ])
    #     wordcloud = WordCloud(stopwords=STOPWORDS,
    #                           background_color=color,
    #                           width=2500,
    #                           height=2000
    #                           ).generate(cleaned_word)
    #     plt.figure(1, figsize=(13, 13))
    #     plt.imshow(wordcloud)
    #     plt.axis('off')
    #     plt.show()

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

    @st.cache
    def train_model(self, tweets):
        # Training the Naive Bayes classifier
        training_set = nltk.classify.apply_features(sentiment.extract_features, tweets)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return classifier

    @st.cache
    def test_model(self, classifier, data):
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

    @st.cache
    def predict_sentiment(self, classifier, data):
        for i in range(len(data)):
            obj = data.iloc[i]['text']
            label = classifier.classify(sentiment.extract_features(obj.split()))

        return label

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
        st.write("\nAccuracy of the model : ", (accuracy_score(actual, predicted)) * 100)

    # def total_sentiment(self, dataset):
    #     actual_number = {}
    #     temp = 'Actual_sentiment'
    #     for i in ['Positive', 'Negative', "Neutral"]:
    #         act = dataset[dataset[temp] == i]
    #         total = len(act)
    #         actual_number[i] = total
    #
    #     predicted_number = {}
    #     for i in ['Positive', 'Negative', "Neutral"]:
    #         pred = dataset[dataset['Predicted_sentiment'] == i]
    #         total = len(pred)
    #         predicted_number[i] = total
    #     return actual_number, predicted_number

    def count_sentiment(self, dataset, column_name):
        self.column_name = column_name
        count = {}
        # temp = 'Actual_sentiment'
        for i in ['Positive', 'Negative', "Neutral"]:
            act = dataset[dataset[self.column_name] == i]
            total = len(act)
            count[i] = total
        return count



    def plt_pie(self, sentiment_count):
        self.count = sentiment_count
        x = sentiment_count.values()
        x = list(x)
        y = sentiment_count.keys()
        y = list(y)
        # Data to plot
        labels = y
        sizes = x
        colors = ['gold', 'yellowgreen', 'lightcoral']
        explode = (0.1, 0.1, 0)  # explode 1st slice
        # Plot
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":

    CSVFileName = "Dataset/Sentiment.csv"
    Trained_Model_File = "Trained_Model/nltk_model/trained_model/"
    predicted_ratio = None
    actual_ratio = None
    prediction = '\nTrain model after selecting \"show test result\".'

    sentiment = sentiment_analysis(CSVFileName)
    # sentiment.choose_ratio()
    dataset, train, test = sentiment.split_data()
    # train_pos, train_neut, train_neg = sentiment.separate(train)
    #     sentiment.wordcloud_draw(train_pos)
    tweets = sentiment.clean_data(train)
    w_features = sentiment.get_word_features(tweets)
    actual_ratio, predicted_ratio = {}, {}

    #   streamlit elements ans variables
    st.title("Sentiment Analysis Application")
    actual_sentiment = list(dataset['sentiment'])
    st.sidebar.header('Data')
    view_data = st.sidebar.checkbox("View initial data")
    data_ratio = st.sidebar.checkbox("View data ratio")

    st.sidebar.subheader("Train Model")
    train_button = st.sidebar.button("Train")
    st.sidebar.subheader("Predict sentiment")
    test_result = st.sidebar.checkbox("Show test result")
    test_button = st.sidebar.button("Test")


    if view_data == True:
        st.write(sentiment.data)

    if data_ratio == True:
        st.write("Ratio of data")
        st.pyplot(sentiment.plt_pie(sentiment.count_sentiment(dataset, 'sentiment')))

    if train_button == True:
        with st.spinner('Training model...'):
            classifier = sentiment.train_model(tweets)
            sentiment.wr_pickle(classifier, Trained_Model_File)
            st.success("Model training successful")

    if test_button == True:
        prediction = None
        with st.spinner('Testing Model'):
            classifier = sentiment.rd_pickle(Trained_Model_File)
            prediction = sentiment.test_model(classifier, test)
            # st.write(prediction)
            st.success("data analysed. sentimented predicted")
            sentiment.acc_score()
            # actual_ratio, predicted_ratio = sentiment.total_sentiment(prediction)
            actual_ratio = sentiment.count_sentiment(prediction, 'Actual_sentiment')
            predicted_ratio = sentiment.count_sentiment(prediction, 'Predicted_sentiment')
            # st.write(actual_ratio)
            # st.write(predicted_ratio)

    if test_result == True:
        # st.warning("Train model after selecting option \"show sentiment.\"")
        # if type(prediction) == 'NoneType':
        #     pass
        # else:
        st.write(prediction)

    data_in = st.text_input("Enter text to predicts its sentiment:")
    data_in = data_in.split("\n")
    txt_area = pd.DataFrame({
        "text":data_in
    })

    if st.button("Predict") == True:
        classifier = sentiment.rd_pickle(Trained_Model_File)
        prediction = sentiment.predict_sentiment(classifier, txt_area)
        if prediction == "Positive":
            st.success("Tweet is Positive")
        elif prediction == "Neutral":
            st.warning("Tweet is Neutral")
        elif prediction == "Negative":
            st.error("Tweet is Negative")
        st.balloons()
