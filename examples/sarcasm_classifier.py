"""Train a compact sarcasm headline classifier with TensorFlow/Keras."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


DATASET = Path(__file__).resolve().parents[1] / "data" / "Sarcasm_Headlines_Dataset.json"
VOCAB_SIZE = 10_000
MAX_LENGTH = 100
EMBEDDING_DIM = 16


with DATASET.open("r", encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream if line.strip()]

sentences = [item["headline"] for item in records]
labels = np.asarray([item["is_sarcastic"] for item in records])
split = int(0.8 * len(sentences))

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences[:split])


def encode(items):
    sequences = tokenizer.texts_to_sequences(items)
    return pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")


train_x = encode(sentences[:split])
test_x = encode(sentences[split:])
train_y = labels[:split]
test_y = labels[split:]

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_x, train_y, epochs=30, validation_data=(test_x, test_y), verbose=2)
