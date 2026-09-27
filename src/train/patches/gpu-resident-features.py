"""Patch openwakeword to hold the training features in VRAM instead of mmap.

The ACAV100M negative features are (5625000, 16, 96) float16 = 17.28 GB, mmap'd
sequentially against 20 GB of RAM: each pass evicts what the next needs, and
training stalls on page faults. It fits in 24 GB of VRAM alongside the model
and activations. NO training dynamics change: same arrays, order, batch
composition - only where the bytes live.

PEAK VRAM IS DURING VALIDATION, NOT TRAINING. Stop the Kokoro containers first
(their two CUDA contexts hold ~2.4 GB; leaving them up is what caused a CUDA
OOM at 37,500 of 50,000 steps, at the first validation):

    docker compose stop kokoro kokoro2

Four edits:

1. `mmap_batch_generator.__init__` - copy each array to a CUDA tensor, in
   chunks (a non-mmap load needs a 17 GB host allocation on a machine with
   ~200 MB free). Deliberately NO fallback to mmap: a silent fallback leaves
   the run looking healthy while delivering none of the benefit - how the
   onnxruntime CPU fallback cost 36 of 83 minutes before anyone noticed.
2. `mmap_batch_generator.__next__` - slice on the GPU. Arrays keep their
   on-disk dtype (ACAV100M float16, the rest float32), so slices are cast to
   float32 before torch.cat, which requires a single dtype; storing ACAV100M
   as float32 would need 34.6 GB. Labels stay on the CPU, cached keyed on the
   per-class row counts (see the __next__ comment).
3. `train.py` - `num_workers=0`: CUDA tensors cannot cross a fork, and with
   the data in VRAM there is no IO left to overlap.
4. `train.py` - validate the false-positive set in chunks of 4096 rather than
   one ~2.76 GiB batch: that spike, not the steady state, runs out of memory;
   the metric is a count over the whole set either way.
"""
import sys

path = sys.argv[1]
target = "train" if path.endswith("train.py") else "data"

with open(path) as f:
    content = f.read()

edits = []

if target == "data":
    # 1. Load each feature array into VRAM, chunked so the host never holds it all.
    old_load = """        self.data = {label: np.load(fl, mmap_mode='r') for label, fl in data_files.items()}"""
    new_load = '''        # GPU-resident features: see patches/gpu-resident-features.py. Chunked, because
        # the host cannot hold a 17 GB array. No fallback - if this cannot allocate,
        # the run must fail rather than quietly crawl on mmap.
        import torch as _torch
        if not _torch.cuda.is_available():
            raise RuntimeError(
                "gpu-resident-features patch is applied but CUDA is unavailable. "
                "Training would silently fall back to a path this patch removed."
            )
        self.data = {}
        for label, fl in data_files.items():
            _src = np.load(fl, mmap_mode='r')
            _dst = _torch.empty(tuple(_src.shape),
                                dtype=getattr(_torch, str(_src.dtype)),
                                device="cuda")
            _chunk = 65536
            for _i in range(0, _src.shape[0], _chunk):
                _dst[_i:_i + _chunk] = _torch.from_numpy(
                    np.ascontiguousarray(_src[_i:_i + _chunk])).cuda()
            print(f"  loaded {label} into VRAM: {tuple(_src.shape)} {_src.dtype} "
                  f"({_dst.element_size() * _dst.nelement() / 1e9:.2f} GB)")
            self.data[label] = _dst
            del _src

        # Label cache; see the __next__ edit for why this is sound.
        self._label_key = None
        self._label_cache = None'''
    edits.append((old_load, new_load))

    # 2. Concatenate on the GPU, and build the labels only when the batch shape
    #    changes rather than on every step.
    old_tail = """                # Make labels for data (following whatever the current shape of `x` is)
                if self.label_files.get(label, None):
                    y_batch = self.labels[label][self.data_counter[label]:self.data_counter[label]+n]
                else:
                    y_batch = [label]*x.shape[0]

                # Transform labels
                if self.label_transform_funcs and self.label_transform_funcs.get(label):
                    y_batch = self.label_transform_funcs[label](y_batch)

                # Add data to batch
                X.append(x)
                y.extend(y_batch)

            return np.vstack(X), np.array(y)"""

    new_tail = """                # Add data to batch. Labels are built after the loop, so they can
                # be reused across steps - see below.
                X.append(x)
                y.append(x.shape[0])

            # `y` holds the per-class row counts. Labels depend on nothing else:
            # label_files is unused here, so each class contributes [key] * n_rows
            # through a stateless transform. n_per_class is fixed, so the label
            # vector is identical on every step EXCEPT the ~1-in-5500 where an
            # array wraps and yields a short slice - hence keying on the row counts.
            # Rebuilding it per step was the single-thread bottleneck once the
            # features moved to VRAM. Caching is only valid without label_files,
            # where labels vary per row; then the key stays None and always misses.
            _cacheable = not self.label_files
            _key = tuple(y)
            if not _cacheable or _key != self._label_key:
                _labels = []
                for (_label, _n), _rows in zip(self.n_per_class.items(), y):
                    if self.label_files.get(_label, None):
                        _batch = self.labels[_label][
                            self.data_counter[_label]:self.data_counter[_label] + _n]
                    else:
                        _batch = [_label] * _rows
                    if self.label_transform_funcs and self.label_transform_funcs.get(_label):
                        _batch = self.label_transform_funcs[_label](_batch)
                    _labels.extend(_batch)
                self._label_cache = np.array(_labels)
                self._label_key = _key if _cacheable else None

            # X holds CUDA tensors of differing dtypes (ACAV100M float16, generated
            # features float32): cast before concatenating. _label_cache is returned
            # by reference; the consumer only reads it (default_convert copies).
            import torch as _torch
            return _torch.cat([_x.to(_torch.float32) for _x in X]), self._label_cache"""
    edits.append((old_tail, new_tail))

else:
    # 3b. Validate in chunks: openwakeword sets batch_size to the whole FP set
    # (~2.76 GiB moved per validation); that spike, not the steady state, is
    # what runs out of memory. 4096 rows is ~48 MiB per step; the metric is a
    # count over the whole set either way.
    old_val = """        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=len(X_val_fp_labels)
        )"""
    new_val = """        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=min(4096, len(X_val_fp_labels))
        )"""
    edits.append((old_val, new_val))

    # 3. CUDA tensors cannot cross a fork, and there is no IO left to overlap.
    old_loader = """        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)"""
    new_loader = """        # num_workers must be 0 with GPU-resident features: CUDA tensors cannot
        # cross a fork, and with the data already in VRAM there is no IO to overlap.
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=0)"""
    edits.append((old_loader, new_loader))

applied = 0
for old, new in edits:
    if old not in content:
        if new.split("\n")[0].strip() in content or "GPU-resident" in content:
            print(f"Already patched: {path}")
            sys.exit(0)
        print(f"ERROR: patch target not found in {path}:\n{old[:120]}")
        sys.exit(1)
    content = content.replace(old, new, 1)
    applied += 1

with open(path, "w") as f:
    f.write(content)

print(f"Patched: {path} ({applied} edit(s))")
