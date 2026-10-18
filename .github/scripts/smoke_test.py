"""Smoke test of an installed fasttext-turbo wheel (run by cibuildwheel)."""
import os
import tempfile

import numpy as np

import fasttext

EXAMPLES = {
    "eng": ["This is a simple English sentence about the weather.",
            "The city is beautiful in the summer."],
    "fra": ["Ceci est une phrase simple sur le temps.",
            "La ville est belle en été."],
    "deu": ["Das ist ein einfacher Satz über das Wetter.",
            "Die Stadt ist im Sommer schön."],
}

with tempfile.TemporaryDirectory() as root:
    train = os.path.join(root, "train.txt")
    with open(train, "w", encoding="utf-8") as f:
        for label, texts in EXAMPLES.items():
            for i in range(60):
                f.write("__label__%s %s %d\n" % (label, texts[i % 2], i))
    for options in ({}, {"crop_prob": 0.5, "crop_min": 3, "crop_max": 20,
                         "norm_prob": 0.5}):
        model = fasttext.train_supervised(input=train, minn=2, maxn=4, dim=16,
                                          epoch=10, lr=0.5, thread=1,
                                          verbose=0, **options)
        labels, probs = model.predict("Die Stadt ist schön", k=2)
        assert len(labels) == 2 and isinstance(probs, np.ndarray), (labels, probs)
        path = os.path.join(root, "model.bin")
        model.save_model(path)
        again = fasttext.load_model(path)
        assert again.predict("Die Stadt ist schön", k=2)[0] == labels
        many = again.predict(["hello world", "bonjour le monde"], k=1)
        assert len(many[0]) == 2
    try:
        fasttext.train_supervised(input=train, crop_prob=1.5, verbose=0)
    except ValueError:
        pass
    else:
        raise AssertionError("crop_prob=1.5 was accepted")

print("fasttext-turbo smoke test passed with numpy", np.__version__)
