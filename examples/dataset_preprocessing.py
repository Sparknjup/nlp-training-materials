"""Load the sarcasm headline JSON Lines file and create padded sequences."""

import json
from pathlib import Path

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


DATASET = Path(__file__).resolve().parents[1] / "data" / "Sarcasm_Headlines_Dataset.json"


def load_headlines(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    sentences = [item["headline"] for item in records]
    labels = [item["is_sarcastic"] for item in records]
    return sentences, labels


sentences, labels = load_headlines(DATASET)
tokenizer = Tokenizer(num_words=10_000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=120, truncating="post", padding="post")

print("samples:", len(labels))
print("tensor shape:", padded.shape)
