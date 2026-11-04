# Prediction on GPUs (or any PyTorch device) for softmax models: FastText.predict(..., device=...).
# PyTorch is needed only here.
#
# The CPU builds each line's features (getFeaturesBytes: words, character n-gram buckets, word n-grams,
# exactly what predict averages) in C++ threads; each device, driven by a thread of its own, averages the
# input rows (an embedding bag), multiplies by the output matrix and takes the top k. Batches go to
# whichever device is free, so faster devices take more of them.
#
# Memory: the model is copied to every device (a clear error if it does not fit). Unless batch_size is
# given, the batch is sized from each device's free memory after the model is loaded; a batch that still
# runs out of memory is split in half and retried. Probabilities come from the top-k logits and the
# logsumexp of the row, so no second (batch x labels) softmax tensor is made.
#
# The result has the CPU predict's format and semantics: labels below the threshold are dropped (the
# threshold is on the probability, as in fastText), probabilities carry fastText's +1e-5 (predict returns
# exp(log(p + 1e-5))), and a line without features gets no prediction. Labels agree with the CPU;
# probabilities differ in the last float digits because the device sums in another order.

import queue
import threading

import numpy as np

_STOP = object()
# 16k-32k lines per batch were fastest on Tesla T4s (GlotLID v4, 2,060 labels); 64k was 13% slower.
MIN_BATCH, MAX_BATCH = 256, 32768
FEATURES_PER_LINE = 1024  # generous: GlotLID v4 averages ~650 rows per CommonLID line


def _torch():
    try:
        import torch
    except ImportError as e:
        raise ImportError("predict(..., device=...) needs PyTorch: pip install fasttext-turbo[gpu] "
                          "(or install torch for your CUDA version)") from e
    return torch


def _oom_types(torch):
    return tuple(t for t in (getattr(torch, "OutOfMemoryError", None),
                             getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)) if t)


class _DeviceModel:
    """A model's input and output matrices on each device, kept for later calls."""

    def __init__(self, model, devices):
        torch = _torch()
        if model.is_quantized():
            raise ValueError("predict on a device needs an unquantized (.bin) model")
        loss = model.f.getArgs().loss.name
        if loss != "softmax":
            raise ValueError("predict on a device supports softmax models only (this one uses %s)" % loss)
        self.devices = [torch.device(d) for d in devices]
        inp = torch.from_numpy(np.ascontiguousarray(model.get_input_matrix(), dtype=np.float32))
        out = torch.from_numpy(np.ascontiguousarray(model.get_output_matrix(), dtype=np.float32))
        self.dim, self.nlab = inp.shape[1], out.shape[0]
        need = (inp.numel() + out.numel()) * 4
        self.bags, self.outs, self.auto_batch = [], [], []
        for d in self.devices:
            try:
                self.bags.append(torch.nn.EmbeddingBag.from_pretrained(inp.to(d), mode="mean", freeze=True))
                self.outs.append(out.to(d))
            except _oom_types(torch) as e:
                raise MemoryError("the model (%.1f GB) does not fit on %s; use a device with more memory "
                                  "or the CPU (device=None)" % (need / 2 ** 30, d)) from e
            self.auto_batch.append(self._batch_for(torch, d))
        self.key = tuple(devices)

    def _batch_for(self, torch, d):
        """Lines per batch so that a batch uses at most a quarter of the device's free memory: per line
        the logits, a logsumexp temporary of the same size and the top k (3 x labels floats), the feature
        ids and the hidden vector."""
        per_line = 4 * (3 * self.nlab + 2 * FEATURES_PER_LINE + 2 * self.dim)
        if d.type == "cuda":
            free, _ = torch.cuda.mem_get_info(d)
        elif d.type == "mps" and hasattr(torch.mps, "recommended_max_memory"):
            # Apple GPUs share the system memory; Metal's recommended working set minus what is in use.
            free = max(0, torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory())
        else:
            free = 2 ** 30  # CPU and other devices: no reliable free-memory query; 1 GB of work memory
        return int(max(MIN_BATCH, min(MAX_BATCH, free // 4 // per_line)))


def _devices(device):
    if isinstance(device, str):
        return [device]
    return list(device)


def predict(model, lines, k, threshold, threads, device, on_unicode_error="strict", batch_size=None):
    torch = _torch()
    oom = _oom_types(torch)
    devices = _devices(device)
    g = getattr(model, "_device_model", None)
    if g is None or g.key != tuple(devices):
        if g is not None:  # free the previous devices' copies before sizing batches on the new ones
            model._device_model = g = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        g = model._device_model = _DeviceModel(model, devices)
    n = len(lines)
    names = model.get_labels(on_unicode_error=on_unicode_error)  # decoded as the CPU predict does
    kk = g.nlab if k <= 0 or k > g.nlab else k
    bs = int(batch_size) if batch_size else min(g.auto_batch)
    starts = list(range(0, n, bs))
    work = queue.Queue(maxsize=2 * len(g.devices) + 2)
    results = [None] * len(starts)
    errors = []

    def produce():
        try:
            for b, s in enumerate(starts):
                if errors:
                    break
                blob = "".join(lines[s:s + bs]).encode("utf-8")
                ids, off = model.f.getFeaturesBytes(blob, threads)
                work.put((b, ids, off))
        except BaseException as e:  # passed on to the caller
            errors.append(e)
        finally:
            for _ in g.devices:
                work.put(_STOP)

    def score(w, ids, off):
        """Top-k probabilities and label ids of the lines in (ids, off); halves on out of memory."""
        dev, bag, out = g.devices[w], g.bags[w], g.outs[w]
        try:
            t_ids = torch.from_numpy(ids).to(dev)
            t_off = torch.from_numpy(off[:-1].astype(np.int32)).to(dev)
            logits = bag(t_ids, t_off) @ out.T
            top, lab = torch.topk(logits, kk, dim=1)
            p = torch.exp(top - torch.logsumexp(logits, dim=1, keepdim=True))
            del logits
            return p.cpu().numpy(), lab.cpu().numpy()
        except oom:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            elif dev.type == "mps":
                torch.mps.empty_cache()
            m = len(off) - 1
            if m <= 1:
                raise
            h = m // 2
            a = score(w, ids[off[0]:off[h]], off[:h + 1] - off[0])
            b = score(w, ids[off[h]:off[m]], off[h:] - off[h])
            return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]])

    def consume(w):
        dev = g.devices[w]
        stream = torch.cuda.Stream(dev) if dev.type == "cuda" else None
        try:
            with torch.inference_mode(), (torch.cuda.stream(stream) if stream else _null()):
                while True:
                    item = work.get()
                    if item is _STOP:
                        return
                    b, ids, off = item
                    p, lab = score(w, ids, off)
                    results[b] = (p, lab, np.diff(off))
        except BaseException as e:
            errors.append(e)
            while work.get() is not _STOP:  # let the producer finish
                pass

    workers = [threading.Thread(target=consume, args=(w,)) for w in range(len(g.devices))]
    for t in workers:
        t.start()
    prod = threading.Thread(target=produce)
    prod.start()
    prod.join()
    for t in workers:
        t.join()
    if errors:
        raise errors[0]

    # Whole batches at once: the top k come sorted, so the labels above the threshold are a prefix of
    # each row; lines without features get no prediction, as on the CPU.
    name_arr = np.array(names, dtype=object)
    all_labels, all_probs = [], []
    for p, lab, counts in results:
        probs = (p + np.float32(1e-5)).astype(np.float32)
        keep = (p >= threshold).sum(axis=1)
        keep[counts == 0] = 0
        lab_rows = name_arr[lab].tolist()
        if (keep == kk).all():
            all_labels.extend(lab_rows)
            all_probs.extend(probs)
        else:
            all_labels.extend(row[:c] for row, c in zip(lab_rows, keep))
            all_probs.extend(row[:c] for row, c in zip(probs, keep))
    return all_labels, all_probs


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
