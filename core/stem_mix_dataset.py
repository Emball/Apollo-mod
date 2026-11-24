"""Read upstream Apollo HDF5 sessions and generate fresh aligned pairs on demand."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


INSTRUMENTS = (
    "bass", "bowed_strings", "drums", "guitar", "other", "other_keys",
    "other_plucked", "percussion", "piano", "vocals", "wind",
)


class StemMixDataset(Dataset):
    def __init__(self, root, num_samples, compressor, sr, segment_sec):
        import h5py

        if sr != 44100 or segment_sec != 3:
            raise ValueError("Upstream stem mixing requires 44.1 kHz and three-second crops")
        if num_samples < 1:
            raise ValueError("datas.stem_mix.num_samples must be positive")
        self.frames = int(sr * segment_sec)
        self.num_samples = num_samples
        self.compressor = compressor
        self.sessions = {}
        for name in (*INSTRUMENTS, "mixture"):
            sources = []
            for path in sorted((Path(root) / name).glob("*.hdf5")):
                with h5py.File(path, "r") as handle:
                    data = handle.get("data")
                    if data is None:
                        continue
                    if data.ndim != 3 or data.shape[1:] != (2, 6 * sr):
                        raise ValueError(f"Unexpected HDF5 session shape in {path}. Use upstream six-second preprocessing.")
                    if data.shape[0]:
                        sources.append((path, data.shape[0], data.shape[-1]))
            if not sources:
                raise ValueError(f"No valid upstream HDF5 sessions in {Path(root) / name}")
            self.sessions[name] = sources
        print(f"Training dataset: {num_samples} fresh stem-mix or original-mix draws per epoch")

    def __len__(self):
        return self.num_samples

    def _crop(self, name):
        import h5py

        path, rows, session_frames = random.choice(self.sessions[name])
        row = random.randrange(rows)
        start = random.randint(0, session_frames - self.frames)
        with h5py.File(path, "r") as handle:
            audio = handle["data"].astype("float32")[row, :, start:start + self.frames]
        return torch.from_numpy(audio)

    def _draw(self):
        if random.random() <= 0.5:
            return self._crop("mixture")
        count = random.randint(1, 8)
        mix = torch.zeros(2, self.frames)
        for category in random.choices(INSTRUMENTS, k=count):
            crop = self._crop(category)
            gain = 10 ** (random.randint(-10, 10) / 20)
            mix.add_(crop, alpha=gain)
        return mix

    def __getitem__(self, index):
        import torch.nn.functional as F

        for _ in range(8):
            hq = self._draw()
            # Keep the same three-second target. Pad only the codec input for alignment.
            pad = self.compressor.pad
            context = F.pad(hq, (pad, pad))
            lq = self.compressor.apply(
                lambda start, length: context[:, start:start + length],
                context.shape[-1], pad, self.frames, "stem_mix", random.getrandbits(64),
            )
            if lq is not None:
                peak = max(hq.abs().max().item(), lq.abs().max().item())
                if peak > 0:
                    return hq / peak, lq / peak
        raise RuntimeError(f"Could not build an aligned stem-mix pair for sample {index} after eight draws")
