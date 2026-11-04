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
        # A list is scored in batches (and threads): the same labels and
        # probabilities as one line at a time, empty lines included.
        lines = [t for texts in EXAMPLES.values() for t in texts] * 300 + ["", "x"]
        one = [again.predict(t, k=3) for t in lines]
        for threads in (1, 4):
            labels_b, probs_b = again.predict(lines, k=3, threads=threads)
            for (l1, p1), l2, p2 in zip(one, labels_b, probs_b):
                assert list(l1) == list(l2), (l1, l2)
                assert np.array_equal(p1, p2), (p1, p2)
        # getFeatures / getFeaturesBytes: the input rows predict averages, so their mean is the
        # sentence vector; device= (when PyTorch and a GPU are there) gives the CPU's labels.
        texts = [t for texts in EXAMPLES.values() for t in texts] + ["", "x"]
        ids, off = again.f.getFeatures([t + "\n" for t in texts], 2)
        ids_b, off_b = again.f.getFeaturesBytes("".join(t + "\n" for t in texts).encode("utf-8"), 3)
        assert np.array_equal(ids, ids_b) and np.array_equal(off, off_b)
        inp = again.get_input_matrix()
        for i, t in enumerate(texts):
            mean = inp[ids[off[i]:off[i + 1]]].mean(axis=0)
            assert np.allclose(mean, again.get_sentence_vector(t), atol=1e-6), t
        try:
            import torch
            gpu = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
        except ImportError:
            gpu = None
        if gpu:
            labels_g, probs_g = again.predict(lines, k=3, threads=2, device=gpu)
            labels_c, probs_c = again.predict(lines, k=3, threads=2)
            assert [list(x) for x in labels_g] == [list(x) for x in labels_c]
            assert max((np.abs(a - b).max() if len(a) else 0.0) for a, b in zip(probs_g, probs_c)) < 1e-4
    try:
        fasttext.train_supervised(input=train, crop_prob=1.5, verbose=0)
    except ValueError:
        pass
    else:
        raise AssertionError("crop_prob=1.5 was accepted")

print("fasttext-turbo smoke test passed with numpy", np.__version__)
