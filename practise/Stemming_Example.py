# import test2
# from test2.classify.util import accuracy
# from test2.classify import NaiveBayesClassifier
#
#
# def form_sent(sent):
#     return {word: True for word in test2.word_tokenize(sent)}
#
#
# ans = form_sent("This is a good book")
# print(ans)
#
# s1 = 'This is a good book'
# s2 = 'This is a awesome book'
# s3 = 'This is a bad book'
# s4 = 'This is a terrible book'
# s5 = 'This is worst book.'
# s6 = 'This book is not awesome'
#
# training_data = [[form_sent(s1), 'pos'], [form_sent(s2), 'pos'], [form_sent(s3), 'neg'], [form_sent(s4), 'neg'], [form_sent(s5), 'neg'], [form_sent(s6), 'neg']]
#
# # for t in training_data:
# #     print(t)
#
# model = NaiveBayesClassifier.train(training_data)
#
# print(model.classify(form_sent('This book is awesome')))
# print(model.classify(form_sent('This is a bad article')))


import nltk
from nltk.corpus import sentiwordnet as swn

words_tagged = [("hate", "n"), ("beautiful", "a")]
ps = nltk.PorterStemmer()
for word, tag in words_tagged:
    print(word)
    word = ps.stem(word)
    concat = word + '.' + tag + '.01'
    print(word)
    ans = swn.senti_synset(concat)  # .pos_score()
    print(ans)
