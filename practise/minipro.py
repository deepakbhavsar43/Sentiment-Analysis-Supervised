import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=10000)
dt = np.concatenate((train_data, test_data), axis=0)
tar = np.concatenate((train_target, test_target), axis=0)


def convert(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


dt = convert(dt)
tar = np.array(tar).astype("float32")
test_x = dt[:9000]
test_y = tar[:9000]
train_x = dt[9000:]
train_y = tar[9000:]
model = models.Sequential()
# Input - Layer
model.add(layers.Dense(50, activation="relu", input_shape=(10000,)))
# Hidden - Layers
model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()
# compiling the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
results = model.fit(
    train_x, train_y,
    epochs=2,
    batch_size=500,
    validation_data=(test_x, test_y)
)
print("Test-Accuracy:", np.mean(results.history["val_acc"]))