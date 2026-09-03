"""Patch openwakeword to hold the training features in VRAM instead of mmap.

Measured on the trainer VM (20 GB RAM, RTX 3090, 4 cores) during the training
stage: GPU utilisation 14%, CPU 37% idle, 7.2 GB of 8 GB swap in use, 213 MB RAM
free. Training is not compute-bound - it stalls on page faults.

The cause is the ACAV100M negative feature file: (5625000, 16, 96) float16, 17.28
GB, against 20 GB of RAM. `mmap_batch_generator` walks it SEQUENTIALLY, 1024 rows
per step, wrapping at the end - so at ~119 steps/s it completes a full pass every
~46 s, roughly 9 passes in 50,000 steps. The working set sits just under total RAM,
so each pass evicts what the next one needs and the kernel swaps to keep up.

17.28 GB fits in 24 GB of VRAM alongside the model and activations, but the margin
is thin and the peak is NOT during training - it is during validation. STOP THE
KOKORO CONTAINERS FIRST; their two CUDA contexts hold ~2.4 GB and are idle by then:

    docker compose stop kokoro kokoro2

Leaving them up is what caused a CUDA OOM at 37,500 of 50,000 steps, at the first
validation, after generation and feature computation had already run.

This changes NO training dynamics: same arrays, same sequential order, same batch
composition. Only where the bytes live changes.

Four edits:

1. `mmap_batch_generator.__init__` - copy each array to a CUDA tensor, in chunks so
   the host never holds more than one chunk. np.load without mmap_mode would need a
   17 GB host allocation on a machine that has 213 MB free.

2. `mmap_batch_generator.__next__` - slice on the GPU. Arrays keep their on-disk
   dtype (ACAV100M is float16; the positive/negative feature files are float32), so
   slices are cast to float32 before concatenation - torch.cat requires a single
   dtype, and storing ACAV100M as float32 would need 34.6 GB. The cast is on a
   1024x16x96 slice, so it is negligible. Labels stay on the CPU: they are a small
   list, and the DataLoader's default_convert turns them into a tensor as before.

3. `train.py` - `num_workers=0`. CUDA tensors cannot cross a fork, so the current
   `num_workers=os.cpu_count()//2` would fail outright. With the data resident in
   VRAM there is no IO left to overlap, so the workers have nothing to do anyway.

4. `train.py` - validate the false-positive set in chunks of 4096 rather than in one
   batch the size of the whole set (~2.76 GiB). That spike, not the steady state, is
   what runs out of memory. The metric is a count over the whole set either way.

There is deliberately NO fallback to the mmap path. A silent fallback would leave
the run looking healthy while delivering none of the benefit - which is exactly how
the onnxruntime CPU fallback cost 36 minutes of an 83-minute run before anyone
noticed the warning. If the allocation fails, this should fail loudly.
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
    new_load = '''        # GPU-resident features: see patches/gpu-resident-features.py. Copied in
        # chunks because the host cannot hold a 17 GB array. No fallback - if this
        # cannot allocate, the run must fail rather than quietly crawl on mmap.
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

            # `y` currently holds the per-class row counts. The labels depend on
            # nothing else: label_files is unused in this pipeline, so each class
            # contributes [key] * n_rows through a stateless transform
            # (lambda x: [1 for i in x] / [0 for i in x]). n_per_class is fixed, so
            # the label vector is identical on every step EXCEPT the roughly 1 in
            # 5500 where an array wraps and yields a short slice - hence keying the
            # cache on the row counts rather than assuming they never change.
            #
            # Rebuilding it per step cost a 1124-element Python list, an np.array and
            # three lambda applications, 100,000 times, on the single thread that
            # became the bottleneck once the features moved to VRAM.
            #
            # Caching is only valid without label_files, where labels vary per row;
            # in that case the key is left as None so the check always misses.
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

            # X holds CUDA tensors of differing dtypes (ACAV100M is float16, the
            # generated features float32), so cast before concatenating.
            # self._label_cache is returned by reference; the consumer only reads it
            # (DataLoader default_convert, then .to(device), which copies).
            import torch as _torch
            return _torch.cat([_x.to(_torch.float32) for _x in X]), self._label_cache"""
    edits.append((old_tail, new_tail))

else:
    # 3b. Validate in chunks rather than one 2.76 GiB allocation.
    #
    # openwakeword sets batch_size to the whole false-positive validation set, so
    # every validation moves ~2.76 GiB to the GPU at once. With 16.6 GiB of resident
    # features that spike is what runs out of memory - and it does so at whatever
    # percentage of training the first validation falls on, after the expensive
    # stages have already completed. 4096 rows is ~48 MiB per step; the metric is a
    # count over the whole set either way, so chunking does not change it.
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
