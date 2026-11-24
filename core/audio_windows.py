"""Bounded reads from the normalized WAV sources used by dataset preparation."""

import soundfile as sf
import torch


class WaveSource:
    def __init__(self, path, sample_rate):
        self.path = path
        info = sf.info(path)
        if info.samplerate != sample_rate:
            raise ValueError(f"{path} must be converted to {sample_rate} Hz before reading")
        self.frames = info.frames

    def read(self, start, frames):
        # Separate file handles let codec workers seek independently.
        with sf.SoundFile(self.path) as source:
            source.seek(min(start, self.frames))
            audio = source.read(frames, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(audio[:, :2].T)
        return tensor.expand(2, -1).contiguous() if tensor.shape[0] == 1 else tensor.contiguous()

    def peak(self, frames):
        peak = 0.0
        for start in range(0, frames, 262144):
            block = self.read(start, min(262144, frames - start))
            peak = max(peak, float(block.abs().max()))
        return peak


def bounded_map(executor, function, items, max_pending):
    """Yield ordered results without submitting the full corpus to the executor."""
    from collections import deque
    from itertools import islice

    items = iter(items)
    pending = deque(executor.submit(function, item) for item in islice(items, max_pending))
    while pending:
        yield pending.popleft().result()
        for item in islice(items, 1):
            pending.append(executor.submit(function, item))
