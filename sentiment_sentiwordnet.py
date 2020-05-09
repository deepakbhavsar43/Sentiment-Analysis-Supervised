from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from nltk import pos_tag, map_tag
# from stemming.porter2 import stem
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import wordcloud as WordCloud

data = pd.read_csv('Dataset/Sentiment.csv')
# Keeping only the neccessary columns
df = data[['text', 'sentiment']]
# print(df.head())

for i in range(len(df)):
    txt = df.loc[i]["text"]
    txt = re.sub(r'@[A-Z0-9a-z_:]+', '', txt)  # replace username-tags
    txt = re.sub(r'^[RT]+', '', txt)  # replace RT-tags
    txt = re.sub('https?://[A-Za-z0-9./]+', '', txt)  # replace URLs
    txt = re.sub("[^a-zA-Z]", " ", txt)  # replace hashtags
    df.at[i, "text"] = txt

lem = nltk.WordNetLemmatizer()
pstem = nltk.PorterStemmer()


# POS-TAGGING AND SENTIMENT SCORE
def pos_senti(df_copy):  # takes
    li_swn = []
    li_swn_pos = []
    li_swn_neg = []
    missing_words = []
    for i in range(len(df_copy.index)):
        text = df_copy.loc[i]['text']
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
                    # Checking if there's a possiblity of lemmatized word be accepted into SWN corpus
                    try:
                        this_word_pos = swn.senti_synset(concat).pos_score()
                        this_word_neg = swn.senti_synset(concat).neg_score()
                    except Exception as e:
                        wor = pstem.stem(word)
                        concat = wor + '.' + tag + '.01'
                        # Checking if there's a possiblity of lemmatized word be accepted
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

        if (pos_total != 0 or neg_total != 0):
            if (pos_total > neg_total):
                li_swn.append(1)
            else:
                li_swn.append(-1)
        else:
            li_swn.append(0)
    df_copy.insert(2, "pos_score", li_swn_pos, True)
    df_copy.insert(3, "neg_score", li_swn_neg, True)
    df_copy.insert(4, "sent_score", li_swn, True)
    return df_copy
    # end-of pos-tagging&sentiment


df_copy = pos_senti(df)

# total_tweets, total_pos, total_neg, total_neu = 0, 0, 0, 0
# for i in range(len(df_copy.index)):
#     temp = df_copy['sent_score']
#     if temp < 0:
#         total_neg += 1
#         total_tweets += 1
#     elif temp > 0:
#         total_pos +=1
#         total_tweets += 1
#     elif temp == 0:
#         total_neu += 1
#         total_tweets += 1

negative = df_copy[df_copy['sent_score'] < 0]
total_neg = len(negative)
positive = df_copy[df_copy['sent_score'] > 0]
total_pos = len(positive)
neutral = df_copy[df_copy['sent_score'] == 0]
total_neu = len(neutral)

# print(f"Total tweets: {df_copy['text'].count()}")
print("Total tweets:", len(df.index))
print("positive tweets:", total_pos)
print("negative tweets:", total_neg)
print("neutral tweets:", total_neu)

# print("\n", df.head())

stop_words = stopwords.words('english')
for i in range(len(df.index)):
    text = df.loc[i]['text']
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
for j in range(len(tokens)):
    tokens[j] = lem.lemmatize(tokens[j])
    tokens[j] = pstem.stem(tokens[j])
tokens_sent = ' '.join(tokens)
df.at[i, "text"] = tokens_sent
print(df)

# Data Visualization
pos_text = ''
neg_text = ''
neut_text = ''
for i in range(len(df_copy.index)):
    if df_copy.loc[i]["sent_score"] == 1:
        pos_text += df_copy.loc[i]["text"]
    elif df_copy.loc[i]["sent_score"] == -1:
        neg_text += df_copy.loc[i]["text"]
    else:
        neut_text += df_copy.loc[i]["text"]

list_text = [pos_text, neg_text, neut_text]
for txt in list_text:
    word_cloud = WordCloud(width=600, height=600, max_font_size=200).generate(txt)
    plt.figure(figsize=(12, 10))  # create a new figure
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

from sklearn.model_selection import train_test_split

x = df.text
y = df.sent_score
SEED = 4
x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.1, random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=SEED)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(decode_error='ignore', lowercase=False, max_features=11)
x_traincv = cv.fit_transform(x_train.values.astype('U'))
top_sum = x_traincv.toarray().sum(axis=0)
top_sum_cv = [top_sum]  # to let pandas know that these are rows
columns_cv = cv.get_feature_names()
x_traincvdf = pd.DataFrame(top_sum_cv, columns=columns_cv)

import operator

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

# Tfidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(decode_error='ignore',lowercase=False,max_features=11)
x_traintf=tf.fit_transform(x_train.values.astype('U'))
top_sum=x_traintf.toarray().sum(axis=0)
top_sum_tf=[top_sum]#to let pandas know that these are rows
columns_tf = tf.get_feature_names()
# print(columns_tf)
x_traintfdf = pd.DataFrame(top_sum_tf,columns=columns_tf)


import operator
dic = {}
for i in range(len(top_sum_tf[0])):
    dic[columns_tf[i]]=top_sum_tf[0][i]
sorted_dic=sorted(dic.items(),reverse=True,key=operator.itemgetter(1))
print(sorted_dic[1:])
bins = [w for w,v in sorted_dic][1:] #slicing to delete the first swachh bharat
freq = [v for w,v in sorted_dic][1:]
from matplotlib import pyplot as plt

plt.figure(figsize=(8,6))
plt.bar(bins,freq)
plt.xlabel('Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top words - Tfidf Vectorizer')
plt.show()

x_train_copy = x_train.copy()
y_train_copy = y_train.copy()
x_val_copy  = x_val.copy()
y_val_copy  = y_val.copy()
# print(x_train_copy)
# print(y_train_copy)

# classifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import datetime
import time

classifiers = [LinearSVC()]
clf_names = ['LinearSVC()']


data=[]
for clf in classifiers:
    before = datetime.datetime.now()
    before = before.strftime("%H:%M:%S")
    start = time.time()
    print(gram)
    cv = CountVectorizer(ngram_range=(1,gram)) #gram3 = 3
    print(gram)
    model = make_pipeline(cv,clf)
    model.fit(x_train_copy.values.astype('U'),y_train_copy.values.astype('U'))##
    labels = model.predict(x_val_copy.values.astype('U'))
    ac = accuracy_score(y_val_copy.values.astype('U'),labels)
    after = datetime.datetime.now()
    after = after.strftime("%H:%M:%S")
    end = time.time()
    hours = int(after[0:2])-int(before[0:2])
    mins = int(after[3:5])-int(before[3:5])
    secs = int(after[6:8])-int(before[6:8])
    time_taken = str(hours)+":"+str(mins)+":"+str(secs)
print(ac)
print(time_taken)


print(df_copy["sent_vec"])
y_ori = df_copy["sent_vec"].values.astype('U')
y_pre = df_copy["sent_score"].values.astype('U')
acc = accuracy_score(y_ori, y_pre)
print("ori vs calculated : ", acc)
acc2 = accuracy_score(y_val_copy.values.astype('U'), labels)
print("ori vs calculated : ", acc2)
