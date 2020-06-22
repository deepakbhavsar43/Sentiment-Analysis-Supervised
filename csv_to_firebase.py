import pandas as pd

dataset = pd.read_csv("Dataset/Sentiment.csv")
count = 0
total_rows = len(dataset)
# print (dataset[["text", "sentiment"]].head())
for i in range(0, total_rows):
    temp = dataset.iloc[i]["text"]
    # try:
    print(count)
    print("-->", temp)
    # except:
    count += 1

