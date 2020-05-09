from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

content = """Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.
In their oldest forms, cakes were modifications of bread, but cakes now cover a wide range of preparations that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards, and pies."""

count_vectorizer = CountVectorizer()
bag_of_words = count_vectorizer.fit_transform(content.splitlines())
print(bag_of_words)
df = pd.DataFrame(bag_of_words.toarray(), columns = count_vectorizer.get_feature_names())
print(df)
