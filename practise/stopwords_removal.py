from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

to_be_removed = set(stopwords.words('english'))
print(to_be_removed)
para = """Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.
In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations 
that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards, 
and pies."""
tokenized_para = word_tokenize(para)
print(tokenized_para)
modified_token_list = [word for word in tokenized_para if not word in to_be_removed]
print(modified_token_list)
