# fastText-turbo

[fastText](https://github.com/facebookresearch/fastText) with faster
prediction and two optional training options. It is a drop-in replacement:
same `import fasttext`, same API, same command line and the same model files,
so models trained with fastText-turbo load in the official fastText and the
other way around.

```bash
pip uninstall fasttext   # both packages install the `fasttext` module
pip install fasttext-turbo
```

## What is different from fastText

- **Faster prediction, same labels.** The output layer is computed eight
  labels at a time (on x86 with SIMD sums), top-k skips the log for labels
  that cannot enter the top k, and subword hashes are computed
  incrementally. Labels are the same as fastText's; probabilities are within
  about 1e-5 on x86 and byte-identical on ARM. On one CPU core, GlotLID v3
  (dimension 256) predicts about 4x faster on x86 and 1.7x on Apple silicon. See
  [Fast Inference for fastText](https://commonglot.github.io/blog/fasttext-turbo.html)
  for how it works and the full benchmarks.
- **Same trained models:** training gives byte-identical models to fastText.
- **Multi-threaded prediction on the command line:**
  `fasttext predict model.bin input.txt k threshold threads`. The output
  keeps the input order.
- **Training on short spans** (`-cropProb`, `-cropMin`, `-cropMax`,
  `-cropUnit char|word`): with probability `cropProb` a training line is
  replaced by a random span of its text, for robustness on short inputs.
- **Training on normalized lines** (`-normProb`): with probability `normProb`
  a training line is lowercased and stripped of punctuation and digits, as
  the CommonLID / OpenLID-v2 input cleaner does.
- **NumPy 1 and 2** are both supported.

Both training options are off by default; then training gives the same model
as fastText. They are not stored in the model file.

```python
import fasttext

model = fasttext.train_supervised(
    input="train.txt", minn=2, maxn=5, dim=64, epoch=3, lr=0.8,
    crop_prob=0.3, crop_min=5, crop_max=20, crop_unit="char", norm_prob=0.3)
labels, probs = model.predict("Das ist ein kleiner Test", k=3)
```

Everything else, including word vectors (`skipgram`, `cbow`), quantization
and autotune, works as documented in
[fastText](https://fasttext.cc/docs/en/python-module.html).

Source: <https://github.com/CommonGlot/fastText-turbo>. MIT license, like
fastText.
