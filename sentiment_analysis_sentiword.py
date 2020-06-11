import re
import nltk
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import streamlit as st
try:
    from wordcloud import WordCloud
except:
    pass
from nltk import pos_tag, map_tag
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from nltk.corpus import sentiwordnet as swn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def read_data():
    csv_data = pd.read_csv(path)
    # Keeping only the neccessary columns
    csv_df = csv_data[['text', 'sentiment']]
    # print(df.head())
    return csv_df


def clean_data(csv_data):
    for i in range(len(dataset)):
        txt = csv_data.loc[i]["text"]
        txt = re.sub(r'@[A-Z0-9a-z_:]+', '', txt)  # replace username-tags
        txt = re.sub(r'^[RT]+', '', txt)  # replace RT-tags
        txt = re.sub('https?://[A-Za-z0-9./]+', '', txt)  # replace URLs
        txt = re.sub("[^a-zA-Z]", " ", txt)  # replace hashtags
        csv_data.at[i, "text"] = txt
    return csv_data


# POS-TAGGING AND SENTIMENT SCORE
def pos_senti(csv_data):  # takes
    li_swn = []
    li_swn_pos = []
    li_swn_neg = []
    li_sent_fact = []
    missing_words = []
    for i in range(len(csv_data.index)):
        text = csv_data.loc[i]['text']
        tokens = nltk.word_tokenize(text)
        tagged_sent = pos_tag(tokens)
        store_it = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_sent]
        # print("Tagged Parts of Speech:",store_it)

        pos_total = 0
        neg_total = 0
        for word, tag in store_it:
            if tag == 'NOUN':
                tag = 'n'
            elif tag == 'VERB':
                tag = 'v'
            elif tag == 'ADJ':
                tag = 'a'
            elif tag == 'ADV':
                tag = 'r'
            else:
                tag = 'nothing'

            if tag != 'nothing':
                concat = word + '.' + tag + '.01'
                try:
                    this_word_pos = swn.senti_synset(concat).pos_score()
                    this_word_neg = swn.senti_synset(concat).neg_score()
                    # print(word,tag,':',this_word_pos,this_word_neg)
                    # except Exception as e:
                    #     print("Error : ",e)
                except Exception as e:
                    wor = lem.lemmatize(word)
                    concat = wor + '.' + tag + '.01'
                    # Checking if there's a possibility of lemmatized word be accepted into SWN corpus
                    try:
                        this_word_pos = swn.senti_synset(concat).pos_score()
                        this_word_neg = swn.senti_synset(concat).neg_score()
                    except Exception as e:
                        wor = pstem.stem(word)
                        concat = wor + '.' + tag + '.01'
                        # Checking if there's a possibility of lemmatized word be accepted
                        try:
                            this_word_pos = swn.senti_synset(concat).pos_score()
                            this_word_neg = swn.senti_synset(concat).neg_score()
                        except:
                            missing_words.append(word)
                            continue
                pos_total += this_word_pos
                neg_total += this_word_neg
        li_swn_pos.append(pos_total)
        li_swn_neg.append(neg_total)

        if pos_total != 0 or neg_total != 0:
            if pos_total > neg_total:
                li_swn.append(1)
            else:
                li_swn.append(-1)
        else:
            li_swn.append(0)
        # factorize sentiment into numeric values
        sent = csv_data.loc[i]['sentiment']
        if sent == 'Neutral':
            li_sent_fact.append(0)
        elif sent == 'Positive':
            li_sent_fact.append(1)
        elif sent == 'Negative':
            li_sent_fact.append(-1)
    csv_data.insert(2, "pos_score", li_swn_pos, True)
    csv_data.insert(3, "neg_score", li_swn_neg, True)
    csv_data.insert(4, "sent_score", li_swn, True)
    csv_data.insert(5, "sent_factorized", li_sent_fact, True)

    return csv_data
    # end-of pos-tagging&sentiment


def rm_stopwords(csv_df):
    stop_words = stopwords.words('english')
    for i in range(len(csv_df.index)):
        text = csv_df.loc[i]['text']
        tokens = nltk.word_tokenize(text)
        #     print(tokens)
        tokens = [word for word in tokens if word not in stop_words]
    #     print(tokens)
    for j in range(len(tokens)):
        tokens[j] = lem.lemmatize(tokens[j])
        tokens[j] = pstem.stem(tokens[j])
    tokens_sent = ' '.join(tokens)
    csv_df.at[i, "text"] = tokens_sent
    return csv_df


def count_top_words():
    # count vectorize top words
    from sklearn.feature_extraction.text import CountVectorizer
    import operator

    count_vect = CountVectorizer(decode_error='ignore', lowercase=False, max_features=11)
    x_traincv = count_vect.fit_transform(x_train.values.astype('U'))
    top_sum = x_traincv.toarray().sum(axis=0)
    top_sum_cv = [top_sum]  # to let pandas know that these are rows
    columns_cv = count_vect.get_feature_names()
    x_traincvdf = pd.DataFrame(top_sum_cv, columns=columns_cv)

    dic = {}
    for i in range(len(top_sum_cv[0])):
        dic[columns_cv[i]] = top_sum_cv[0][i]
    sorted_dic = sorted(dic.items(), reverse=True, key=operator.itemgetter(1))
    print(sorted_dic[1:])
    bins = [w for w, v in sorted_dic][1:]  # slicing to delete the first swachh bharat
    freq = [v for w, v in sorted_dic][1:]
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.bar(bins, freq)
    plt.xlabel('Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top words - Count Vectorizer')
    plt.show()


def count_tfidf_vectorizer():
    # Tfidf Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer(decode_error='ignore', lowercase=False, max_features=11)
    x_traintf = tf.fit_transform(x_train.values.astype('U'))
    top_sum = x_traintf.toarray().sum(axis=0)
    top_sum_tf = [top_sum]  # to let pandas know that these are rows
    columns_tf = tf.get_feature_names()
    # print(columns_tf)
    x_traintfdf = pd.DataFrame(top_sum_tf, columns=columns_tf)

    import operator
    dic = {}
    for i in range(len(top_sum_tf[0])):
        dic[columns_tf[i]] = top_sum_tf[0][i]
    sorted_dic = sorted(dic.items(), reverse=True, key=operator.itemgetter(1))
    print(sorted_dic[1:])
    bins = [w for w, v in sorted_dic][1:]  # slicing to delete the first swachh bharat
    freq = [v for w, v in sorted_dic][1:]
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.bar(bins, freq)
    plt.xlabel('Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top words - Tfidf Vectorizer')
    plt.show()


def classify_text():
    positive_text = ''
    negative_text = ''
    neutral_text = ''
    for i in range(len(df_copy.index)):
        if df_copy.loc[i]["sent_score"] == 1:
            positive_text += df_copy.loc[i]["text"]
        elif df_copy.loc[i]["sent_score"] == -1:
            negative_text += df_copy.loc[i]["text"]
        else:
            neutral_text += df_copy.loc[i]["text"]
    return positive_text, negative_text, neutral_text


def visualize_words(txt):
    word_cloud = WordCloud(width=900, height=600, max_font_size=200).generate(txt)
    plt.figure(figsize=(12, 10))  # create a new figure
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def count_sentiment(dataset, column_name):
    # column_name = column_name
    count = {}
    # temp = 'Actual_sentiment'
    for i in [-1, 0, 1]:
        act = dataset[dataset[column_name] == i]
        total = len(act)
        count[i] = total
    return count

def plt_pie(sentiment_count):
    count = sentiment_count
    x = sentiment_count.values()
    x = list(x)
    y = sentiment_count.keys()
    y = list(y)
    # Data to plot
    labels = y
    st.write(labels)
    sizes = x
    colors = ['gold', 'yellowgreen', 'lightcoral']
    explode = (0.1, 0.1, 0)  # explode 1st slice
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()

def wr_pickle(train, model):
    # creating pickle file
    outfile = open(model, 'wb')
    # saving trained model in pickle file
    pickle.dump(train, outfile)
    outfile.close()


def rd_pickle(model):
    # opening pickle file
    infile = open(model, 'rb')
    # loading the saved model into variable newTraining
    newTraining = pickle.load(infile)
    return newTraining


if __name__ == "__main__":
    path = 'Dataset/Dataset.csv'
    Trained_Model_File = 'Trained_Model/sentiwordnet_model/trained_model'
    dataset = read_data()
    df = clean_data(dataset)
    lem = nltk.WordNetLemmatizer()
    pstem = nltk.PorterStemmer()
    df_copy = pos_senti(df)
    df = rm_stopwords(df)

    # streamlit elements ans variables
    st.title("Sentiment Analysis Application (Semi-Supervised)")
    st.header("Semi-Supervised")
    actual_sentiment = list(dataset['sentiment'])
    st.sidebar.header('Dataset')
    view_data = st.sidebar.checkbox("View initial data")
    data_ratio = st.sidebar.checkbox("View data ratio")
    # pos_data = st.sidebar.checkbox("View positive text")
    # neu_data = st.sidebar.checkbox("View neutral text")
    # neg_data = st.sidebar.checkbox("View negative text")
    # data_ratio = st.sidebar.checkbox("View data ratio")

    st.sidebar.subheader("Train Model")
    train_button = st.sidebar.button("Train")
    st.sidebar.subheader("Predict sentiment")
    # test_result = st.sidebar.checkbox("Show test result")
    test_button = st.sidebar.button("Test")

    if view_data:
        st.write(df_copy)

    if data_ratio == True:
        st.write("Ratio of data")
        st.pyplot(plt_pie(count_sentiment(dataset, 'sent_score')))

    # visualize words
    pos_text, neg_text, neut_text = classify_text()

    # if pos_data:
    #     st.pyplot(visualize_words(pos_text))
    #
    # if neu_data:
    #     st.pyplot(visualize_words(neg_text))
    #
    # if neg_data:
    #     st.pyplot(visualize_words(neut_text))

    # split data
    x = df.text
    y = df.sent_score
    SEED = 4
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.1, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=SEED)

    if train_button == True:
        with st.spinner('Training model...'):
            classifiers = [LinearSVC()]
            clf_names = ['LinearSVC()']
            gram = 3
            data = []
            for clf in classifiers:
                cv = CountVectorizer(ngram_range=(1, gram))  # gram3 = 3
                model = make_pipeline(cv, clf)
                mlmodel = model.fit(x_train.values.astype('U'), y_train.values.astype('U'))
                wr_pickle(mlmodel, Trained_Model_File)
        st.success("Model training successful")

    if test_button == True:
        with st.spinner('Testing model...'):
            model = rd_pickle(Trained_Model_File)
            labels = model.predict(x_val.values.astype('U'))
            ac = accuracy_score(y_val.values.astype('U'), labels)
            st.success("data analysed. sentimented predicted")
            st.write("\nAccuracy of the model : ", ac*100)

    txt_area = st.text_input("Enter text to predicts its sentiment:")
    txt_area = txt_area.split("\n")
    # data_in = np.array(data_in)
    txt_area = pd.Series(np.array(txt_area))
    # st.write(type(txt_area))

    if st.button("Predict"):
        model = rd_pickle(Trained_Model_File)
        prediction = model.predict(txt_area)
        st.write(prediction[0])
        if prediction[0] == "1":
            st.success("Tweet is Positive")
        elif prediction == "0":
            st.warning("Tweet is Neutral")
        elif prediction == "-1":
            st.error("Tweet is Negative")
        st.balloons()

    # df = pd.DataFrame({
    #     "full_text": ["I am Deepak. I have developed this project during my internship. i am so happy."]
    # })
    # print(accuracy_score(df_copy["sent_score"], df_copy["sent_factorized"]))
