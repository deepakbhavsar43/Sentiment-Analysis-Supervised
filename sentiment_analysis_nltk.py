import nltk
import pickle
import pandas as pd
import streamlit as st
from csv import DictWriter
import matplotlib.pyplot as plt
from firebase import firebase
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# from userInput import *
from wordcloud import WordCloud, STOPWORDS


class sentiment_analysis:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        # Keeping only the neccessary columns
        self.data = self.data[['text', 'sentiment']]
        # self.data = sentiment_analysis.choose_ratio(self.data)

    # to use only small amount of data for training and testing model
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

    # Splitting the dataset into train and test set
    def split_data(self):
        self.train, self.test = train_test_split(self.data, test_size=0.1)
        return self.train, self.test

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

    def split_dataset(self, threshold=160):
        text_train, sent_train = [], []
        text_test, sent_test = [], []
        for s in ['Negative', 'Neutral', 'Positive']:
            temp = self.data[self.data['sentiment'] == s]
            length = len(temp)
            for i in range(0, length):
                txt = temp.iloc[i]['text']
                sent = temp.iloc[i]['sentiment']
                if i <= (length - threshold):
                    text_train.append(txt)
                    sent_train.append(sent)
                else:
                    text_test.append(txt)
                    sent_test.append(sent)

            train = pd.DataFrame({
                "text":text_train,
                "sentiment":sent_train
            })

            test = pd.DataFrame({
                "text": text_test,
                "sentiment": sent_test
            })
        return train, test

    # seperates all the positive negative and neutral tweets
    def separate(self, data):
        positive = data[data['sentiment'] == 'Positive']
        positive = positive['text']
        neutral = data[data['sentiment'] == 'Neutral']
        neutral = neutral['text']
        negative = data[data['sentiment'] == 'Negative']
        negative = negative['text']
        return positive, neutral, negative

    # removes mentions, links, RT and hashtags from tweets
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

    # Training the Naive Bayes classifier
    @st.cache
    def train_model(self, tweets):
        training_set = nltk.classify.apply_features(sentiment.extract_features, tweets)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return classifier

    # To test model on test data
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

    # to make predictions on unseen data
    @st.cache
    def predict_sentiment(self, classifier, data):
        for i in range(len(data)):
            obj = data.iloc[i]['text']
            label = classifier.classify(sentiment.extract_features(obj.split()))

        return label

    # To store the trained model in pickel file
    def wr_pickle(self, train, model):
        # creating pickle file
        outfile = open(model, 'wb')
        # saving trained model in pickle file
        pickle.dump(train, outfile)
        outfile.close()

    # To read stored model in pickel file
    def rd_pickle(self, model):
        # opening pickle file
        infile = open(model, 'rb')
        # loading the saved model into variable newTraining
        self.newTraining = pickle.load(infile)
        return self.newTraining

    # to calculate the accuracy of the model
    def acc_score(self, result):
        self.result = result
        actual = self.result["Actual_sentiment"]
        predicted = self.result["Predicted_sentiment"]
        st.write("\nAccuracy of the model : ", (accuracy_score(actual, predicted)) * 100)

    #  To calculate total number of positive, negative and neutral tweet
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

    def append_dict_as_row(self, file_name, dict_of_elem, field_names):
        # Open file in append mode
        with open(file_name, 'a+') as write_obj:
            # Create a writer object from csv module
            dict_writer = DictWriter(write_obj, fieldnames=field_names)
            # Add dictionary as wor in the csv
            dict_writer.writerow(dict_of_elem)


def find_min(dict):
    dict_values = dict.values()
    return min(dict_values)


if __name__ == "__main__":

    csv_dataset = "Dataset/Sentiment.csv"
    Trained_Model_File = "Trained_Model/nltk_model/trained_model"
    predicted_ratio = None
    actual_ratio = None
    prediction = '\nTrain model after selecting \"show test result\".'
    # prediction = None

    sentiment = sentiment_analysis(csv_dataset)
    # sentiment.choose_ratio()
    threshold = 160
    # sentiment.wordcloud_draw(train_pos)
    train, test = sentiment.split_data()
    # length of train data and test data
    # st.write(len(train), len(test))
    tweets = sentiment.clean_data(train)
    w_features = sentiment.get_word_features(tweets)
    actual_ratio, predicted_ratio = {}, {}

    #   streamlit elements ans variables
    st.title("Sentiment Analysis Application")
    st.sidebar.info("Supervised Model.")
    # st.info("This model use supervised machine learning technique.")

    st.sidebar.subheader('Navigate to')
    page = st.sidebar.selectbox("", ["Training and Testing", "Prediction"])
    actual_sentiment = list(sentiment.data['sentiment'])

    if page == "Training and Testing":
        st.sidebar.subheader('Data')
        view_data = st.sidebar.checkbox("View dataset")
        sentiment_ratio = st.sidebar.checkbox("View sentiment ratio")

        st.sidebar.subheader("Train Model")
        view_traindata = st.sidebar.checkbox("Training Data")
        train_button = st.sidebar.button("Train")
        st.sidebar.subheader("Test Model")
        view_testdata = st.sidebar.checkbox("Testing Data")
        test_result = st.sidebar.checkbox("Show test result")
        test_button = st.sidebar.button("Test")
        if view_data == True:
            st.write("Dataset :",sentiment.data)

        if sentiment_ratio == True:
            st.write("Ratio of sentiment :")
            st.pyplot(sentiment.plt_pie(sentiment.count_sentiment(sentiment.data, 'sentiment')))

        if view_traindata:
            st.write("Training data :", train)

        if view_testdata:
            st.write("Testing data :", test)

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
                sentiment.acc_score(prediction)
                actual_ratio = sentiment.count_sentiment(prediction, 'Actual_sentiment')
                predicted_ratio = sentiment.count_sentiment(prediction, 'Predicted_sentiment')
                # st.write(actual_ratio)
                # st.write(predicted_ratio)

            if test_result == True:
                st.write(prediction)

    if page == "Prediction":
        data_in = st.text_input("Enter text to predicts its sentiment:")
        data_in = data_in.split("\n")
        txt_area = pd.DataFrame({
            "text": data_in
        })

        if st.button("Predict") == True:
            classifier = sentiment.rd_pickle(Trained_Model_File)
            prediction = sentiment.predict_sentiment(classifier, txt_area)
            pred_data = {
                "text": data_in[0],
                "sentiment": prediction
            }
            try:
                firebase = firebase.FirebaseApplication("https://sentiment-analysis-nltk.firebaseio.com/", None)
                firebase.post('/sentiment-analysis-nltk/userPredictions', pred_data)
            except:
                st.warning("Connection Error : could not able to connect to the internet. check connection.")
            if prediction == "Positive":
                st.success("Tweet is Positive")
            elif prediction == "Neutral":
                st.warning("Tweet is Neutral")
            elif prediction == "Negative":
                st.error("Tweet is Negative")
            st.balloons()
