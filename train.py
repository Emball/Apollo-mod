import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision("high")  # TF32 — faster, negligible quality difference
import hydra
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.models.apollo
from look2hear.utils import RankedLogger, instantiate, print_only
import random
import shutil
import warnings
warnings.filterwarnings("ignore")


# ── Constants mirrored from preprocess_pairs.py ───────────────────────────────
_SR           = 44100
_CHUNK_SEC    = 3
_OVERLAP      = 0.5
_CHUNK_SAMPLES = int(_CHUNK_SEC * _SR)
_HOP_SAMPLES   = int(_CHUNK_SAMPLES * (1 - _OVERLAP))
_SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".aac", ".ogg", ".m4a"}
_N_EVAL_PAIRS   = 3   # how many chunk pairs to reserve for eval when eval dir is empty


# ---------------------------------------------------------------------------
# Data preparation — runs before training, skips gracefully if already done
# ---------------------------------------------------------------------------

def _load_wav_stereo(path: str):
    """Load any supported audio file, resample to _SR, force stereo."""
    import torchaudio
    wav, sr = torchaudio.load(path)
    if sr != _SR:
        wav = torchaudio.functional.resample(wav, sr, _SR)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def _slice_and_save(lq_wav, hq_wav, stem: str, lq_out: str, hq_out: str) -> list:
    """Slice a pair into overlapping chunks, save both sides, return filenames."""
    import torchaudio
    min_len = min(lq_wav.shape[-1], hq_wav.shape[-1])
    lq_wav  = lq_wav[:, :min_len]
    hq_wav  = hq_wav[:, :min_len]

    saved = []
    start = 0
    idx   = 0
    while start + _CHUNK_SAMPLES <= min_len:
        fname = f"{stem}_{idx:04d}.wav"
        torchaudio.save(os.path.join(lq_out, fname), lq_wav[:, start:start + _CHUNK_SAMPLES], _SR)
        torchaudio.save(os.path.join(hq_out, fname), hq_wav[:, start:start + _CHUNK_SAMPLES], _SR)
        saved.append(fname)
        start += _HOP_SAMPLES
        idx   += 1
    return saved


def _has_wav_pairs(lq_dir: str, hq_dir: str) -> bool:
    """Return True if both dirs exist and share at least one matching stem."""
    if not (os.path.isdir(lq_dir) and os.path.isdir(hq_dir)):
        return False
    lq_stems = {os.path.splitext(f)[0] for f in os.listdir(lq_dir) if f.endswith(".wav")}
    hq_stems = {os.path.splitext(f)[0] for f in os.listdir(hq_dir) if f.endswith(".wav")}
    return bool(lq_stems & hq_stems)


def _count_wav_pairs(lq_dir: str, hq_dir: str) -> int:
    if not (os.path.isdir(lq_dir) and os.path.isdir(hq_dir)):
        return 0
    lq_stems = {os.path.splitext(f)[0] for f in os.listdir(lq_dir) if f.endswith(".wav")}
    hq_stems = {os.path.splitext(f)[0] for f in os.listdir(hq_dir) if f.endswith(".wav")}
    return len(lq_stems & hq_stems)


def prepare_data(cfg: DictConfig) -> None:
    """
    Auto-preprocessing pipeline called before training.

    1. If chunks/LQ and chunks/HQ already contain matched pairs → skip chunking.
    2. Otherwise, look for raw paired audio in data/LQ and data/HQ and chunk it.
    3. If eval/LQ and eval/HQ are empty / missing, randomly pull _N_EVAL_PAIRS
       chunk pairs out of chunks/ into eval/ (removed from training set).
    """
    chunks_dir = cfg.datas.train_dir   # e.g. ./chunks
    eval_dir   = cfg.datas.eval_dir    # e.g. ./eval
    data_dir   = os.path.join(os.path.dirname(chunks_dir), "data")  # ./data

    chunks_lq = os.path.join(chunks_dir, "LQ")
    chunks_hq = os.path.join(chunks_dir, "HQ")
    eval_lq   = os.path.join(eval_dir,   "LQ")
    eval_hq   = os.path.join(eval_dir,   "HQ")
    data_lq   = os.path.join(data_dir,   "LQ")
    data_hq   = os.path.join(data_dir,   "HQ")

    sep = "=" * 58

    # ── Step 1: Chunking ─────────────────────────────────────────────────────
    if _has_wav_pairs(chunks_lq, chunks_hq):
        n = _count_wav_pairs(chunks_lq, chunks_hq)
        print_only(f"[data] chunks/ already contains {n} pairs — skipping preprocessing.")
    else:
        # Look for raw data
        if not _has_wav_pairs(data_lq, data_hq):
            print_only(
                f"[data] WARNING: No chunks found in {chunks_dir} and no raw data found in {data_dir}. "
                f"Make sure one of these exists before training."
            )
            return

        os.makedirs(chunks_lq, exist_ok=True)
        os.makedirs(chunks_hq, exist_ok=True)

        # Collect matched pairs from data/
        lq_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(data_lq)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS
        }
        hq_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(data_hq)
            if os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS
        }
        matched = sorted(set(lq_files) & set(hq_files))
        unmatched_lq = set(lq_files) - set(hq_files)
        unmatched_hq = set(hq_files) - set(lq_files)
        if unmatched_lq:
            print_only(f"[data] WARNING: LQ files with no HQ match (skipping): {sorted(unmatched_lq)}")
        if unmatched_hq:
            print_only(f"[data] WARNING: HQ files with no LQ match (skipping): {sorted(unmatched_hq)}")
        if not matched:
            print_only(f"[data] ERROR: No matched pairs in {data_dir}. Cannot preprocess.")
            return

        print_only("\n[data] " + sep)
        print_only(f"[data] Auto-preprocessing {len(matched)} source pairs → {chunks_dir}")
        print_only(f"[data] {sep}\n")

        total = 0
        for stem in matched:
            lq_wav = _load_wav_stereo(os.path.join(data_lq, lq_files[stem]))
            hq_wav = _load_wav_stereo(os.path.join(data_hq, hq_files[stem]))
            saved  = _slice_and_save(lq_wav, hq_wav, stem, chunks_lq, chunks_hq)
            print_only(f"[data]   {stem}: {len(saved)} chunks")
            total += len(saved)

        print_only(f"[data] Chunking complete — {total} chunk pairs written to {chunks_dir}\n")

    # ── Step 2: Eval bootstrap ────────────────────────────────────────────────
    if _has_wav_pairs(eval_lq, eval_hq):
        n = _count_wav_pairs(eval_lq, eval_hq)
        print_only(f"[data] eval/ already contains {n} pairs — skipping eval bootstrap.")
        return

    # Eval is empty — pick random chunks and move them into eval/
    all_lq_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(chunks_lq) if f.endswith(".wav") and "aug_" not in f
    }
    all_hq_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(chunks_hq) if f.endswith(".wav") and "aug_" not in f
    }
    available = sorted(all_lq_stems & all_hq_stems)

    if len(available) < _N_EVAL_PAIRS:
        print_only(
            f"[data] WARNING: Only {len(available)} original chunk pairs available, "
            f"need {_N_EVAL_PAIRS} for eval. Using all of them — consider adding more data.")
        chosen = available
    else:
        chosen = random.sample(available, _N_EVAL_PAIRS)

    os.makedirs(eval_lq, exist_ok=True)
    os.makedirs(eval_hq, exist_ok=True)

    print_only("\n[data] eval/ is empty — bootstrapping " + str(len(chosen)) + " eval pairs from chunks/")
    for stem in chosen:
        lq_src = os.path.join(chunks_lq, f"{stem}.wav")
        hq_src = os.path.join(chunks_hq, f"{stem}.wav")
        shutil.move(lq_src, os.path.join(eval_lq, f"{stem}.wav"))
        shutil.move(hq_src, os.path.join(eval_hq, f"{stem}.wav"))
        print_only(f"[data]   moved {stem}.wav → eval/")

        # Remove augmented variants of this chunk from training
        for fname in os.listdir(chunks_lq):
            if fname.startswith(stem + "_aug_"):
                os.remove(os.path.join(chunks_lq, fname))
                lq_aug_hq = os.path.join(chunks_hq, fname)
                if os.path.exists(lq_aug_hq):
                    os.remove(lq_aug_hq)

    remaining = _count_wav_pairs(chunks_lq, chunks_hq)
    print_only(f"[data] eval bootstrap complete. {remaining} chunk pairs remain in training set.\n")


def freeze_early_layers(model, n_layers_to_freeze=4):
    """
    Freeze the band-split front-end (BN) and first N BSNet layers.
    Default of 4 keeps VRAM and backprop cost manageable on the universal
    model (feature_dim=384) with an 11 GB card — only layers 4-5 and the
    output heads are trained, which is where band reconstruction happens
    and where codec-specific adaptation matters most.
    """
    # Freeze band normalization and bottleneck front-end
    for param in model.BN.parameters():
        param.requires_grad = False

    # Freeze first N layers of the BSNet stack
    for i, layer in enumerate(model.net):
        if i < n_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total  = sum(p.numel() for p in model.parameters())
    print_only(f"Frozen {frozen:,} / {total:,} parameters ({100*frozen/total:.1f}%)")


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Auto-preprocess raw data and bootstrap eval if needed
    prepare_data(cfg)

    # Instantiate datamodule
    print_only(f"Instantiating datamodule <{cfg.datas._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datas)

    # Load pretrained Apollo model
    feature_dim = cfg.model.get("feature_dim", 256)
    local_path = cfg.get("weights_path", None)

    if local_path:
        # Local file provided — support both .pth (from_pretrain format)
        # and .ckpt (Lightning checkpoint, state dict nested under "state_dict" key)
        print_only(f"Loading weights from local file: {local_path}")
        if local_path.endswith(".ckpt"):
            ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
            raw = ckpt["state_dict"]

            # Handle two possible formats:
            # 1. Lightning checkpoint — keys prefixed with "audio_model."
            # 2. UVR-style export or third-party ckpt — bare keys, no prefix
            if any(k.startswith("audio_model.") for k in raw.keys()):
                model_state = {k.replace("audio_model.", ""): v
                               for k, v in raw.items() if k.startswith("audio_model.")}
                print_only("Detected Lightning checkpoint format (audio_model. prefix)")
            else:
                model_state = raw
                print_only("Detected bare state dict format (no prefix)")

            model = look2hear.models.apollo.Apollo(sr=44100, win=20,
                                                   feature_dim=feature_dim, layer=6)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            if missing:
                print_only(f"Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print_only(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            if not missing:
                print_only("All keys loaded successfully.")
        else:
            # .pth — standard from_pretrain format
            model = look2hear.models.BaseModel.from_pretrain(
                local_path, sr=44100, win=20, feature_dim=feature_dim, layer=6
            )
        print_only("Weights loaded from local file.")
    else:
        # No local path — download from HuggingFace
        # from_pretrain() does a raw torch.load() and does NOT use the HF hub API.
        print_only("Downloading pretrained Apollo weights from HuggingFace...")
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id="JusperLee/Apollo",
            filename="pytorch_model.bin",
        )
        print_only(f"Weights cached at: {weights_path}")
        model = look2hear.models.BaseModel.from_pretrain(
            weights_path, sr=44100, win=20, feature_dim=feature_dim, layer=6
        )
        print_only("Pretrained weights loaded.")

    # Freeze early layers for fine-tuning
    # BN + layers 0-3 frozen; only layers 4-5 and output heads are trainable.
    # The uni model (feature_dim=384) is too wide to unfreeze more than this
    # on an 11 GB card without a severe throughput hit.
    freeze_early_layers(model, n_layers_to_freeze=4)

    # Instantiate discriminator fresh — learns your artifact type from scratch
    print_only(f"Instantiating Discriminator <{cfg.discriminator._target_}>")
    discriminator = hydra.utils.instantiate(cfg.discriminator)

    # Instantiate optimizers
    print_only(f"Instantiating optimizers")
    base_lr = cfg.optimizer_g.lr
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.AdamW(
        params=trainable_params,
        lr=base_lr,
        weight_decay=cfg.optimizer_g.get("weight_decay", 0.01),
    )
    optimizer_d = hydra.utils.instantiate(cfg.optimizer_d, params=discriminator.parameters())

    # Instantiate schedulers
    scheduler_g = hydra.utils.instantiate(cfg.scheduler_g, optimizer=optimizer_g)
    scheduler_d = hydra.utils.instantiate(cfg.scheduler_d, optimizer=optimizer_d)

    # Instantiate losses
    print_only(f"Instantiating losses")
    loss_g = hydra.utils.instantiate(cfg.loss_g)
    loss_d = hydra.utils.instantiate(cfg.loss_d)
    losses = {"g": loss_g, "d": loss_d}

    # Instantiate metrics
    print_only(f"Instantiating metrics <{cfg.metrics._target_}>")
    metrics = hydra.utils.instantiate(cfg.metrics)

    # Instantiate system
    print_only(f"Instantiating system <{cfg.system._target_}>")
    system: LightningModule = hydra.utils.instantiate(
        cfg.system,
        model=model,
        discriminator=discriminator,
        loss_func=losses,
        metrics=metrics,
        optimizer=[optimizer_g, optimizer_d],
        scheduler=[scheduler_g, scheduler_d]
    )

    # Point the system at the sample output directory so validation saves audio
    sample_dir = os.path.join(cfg.exp.dir, cfg.exp.name, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    system.sample_output_dir = sample_dir
    print_only(f"Validation audio samples will be saved to: {sample_dir}")

    def _process_audio_file(lq_wav, model, device, chunk_samples=_CHUNK_SAMPLES):
        """
        Run a full stereo waveform through the model using overlap-add chunking.
        Returns the restored waveform as a CPU tensor (2, N).
        """
        lq = lq_wav.to(device)
        total_samples = lq.shape[-1]
        hop = chunk_samples // 2
        output = torch.zeros(2, total_samples, device=device)
        weight = torch.zeros(total_samples, device=device)
        window = torch.hann_window(chunk_samples, device=device)

        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = lq[:, start:end]

            pad = chunk_samples - chunk.shape[-1]
            if pad > 0:
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            scale = chunk.abs().max()
            if scale > 1e-8:
                chunk_norm = chunk / scale
            else:
                chunk_norm = chunk
                scale = torch.tensor(1.0, device=device)

            out_chunk = model(chunk_norm.unsqueeze(0)).squeeze(0) * scale

            actual_len = end - start
            output[:, start:end] += out_chunk[:, :actual_len] * window[:actual_len]
            weight[start:end] += window[:actual_len]

            start += hop

        weight = weight.clamp(min=1e-8)
        output = (output / weight.unsqueeze(0)).clamp(-1.0, 1.0).cpu()
        return output

    class TestFolderCallback(pl.Callback):
        """
        After each validation epoch, processes every audio file found in
        test_dir and saves the restored output to:

            <test_output_dir>/epoch_<NNNN>/<original_filename>.wav

        Drop any audio file into test_dir and it will be processed each epoch.
        Supports: .wav .flac .mp3 .aac .ogg .m4a
        """
        def __init__(self, test_dir: str, output_dir: str, sr: int = 44100):
            self.test_dir = test_dir
            self.output_dir = output_dir
            self.sr = sr
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            print_only(f"[test] Watching folder: {test_dir}")
            print_only(f"[test] Outputs will be saved to: {output_dir}/epoch_NNNN/")

        def _find_audio_files(self):
            found = []
            for fname in sorted(os.listdir(self.test_dir)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in _SUPPORTED_EXTS:
                    found.append(os.path.join(self.test_dir, fname))
            return found

        @torch.no_grad()
        def on_validation_epoch_end(self, trainer, pl_module):
            import torchaudio
            files = self._find_audio_files()
            if not files:
                print_only(f"[test] No audio files in {self.test_dir} — skipping.")
                return

            epoch = trainer.current_epoch
            epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch:04d}")
            os.makedirs(epoch_dir, exist_ok=True)

            device = next(pl_module.parameters()).device
            model = pl_module.audio_model
            model.eval()

            for fpath in files:
                fname = os.path.basename(fpath)
                stem = os.path.splitext(fname)[0]

                try:
                    wav, file_sr = torchaudio.load(fpath)
                    if file_sr != self.sr:
                        wav = torchaudio.functional.resample(wav, file_sr, self.sr)
                    if wav.shape[0] == 1:
                        wav = wav.repeat(2, 1)
                    elif wav.shape[0] > 2:
                        wav = wav[:2]

                    restored = _process_audio_file(wav, model, device)
                    out_path = os.path.join(epoch_dir, f"{stem}.wav")
                    torchaudio.save(out_path, restored, self.sr)
                    print_only(f"[test] epoch {epoch:04d} | {fname} → {out_path}")

                except Exception as e:
                    print_only(f"[test] ERROR processing {fname}: {e}")

            model.train()

    class UVRExportCallback(pl.Callback):
        """After each checkpoint save, write a UVR-compatible version to uvr_exports/."""
        def __init__(self, export_dir, feature_dim=256):
            self.export_dir = export_dir
            self.feature_dim = feature_dim
            os.makedirs(export_dir, exist_ok=True)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            epoch = trainer.current_epoch
            val_loss = trainer.callback_metrics.get("val_loss", 0)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_name = f"epoch={epoch:04d}-val_loss={val_loss:.4f}-{ts}_uvr.ckpt"
            out_path = os.path.join(self.export_dir, out_name)

            raw_state = checkpoint["state_dict"]
            model_state = {
                k.replace("audio_model.", ""): v
                for k, v in raw_state.items()
                if k.startswith("audio_model.")
            }
            uvr_dict = {
                "model_name": "Apollo",
                "state_dict": model_state,
                "model_args": {
                    "sr": 44100,
                    "win": 20,
                    "feature_dim": self.feature_dim,
                    "layer": 6,
                },
                "infos": {
                    "training_epoch": epoch,
                    "val_loss": float(val_loss),
                },
            }
            torch.save(uvr_dict, out_path)
            print_only(f"UVR export saved: {out_path}")

    # Instantiate callbacks
    callbacks: List[Callback] = []

    # Test folder — processes all audio files in test_dir each val epoch
    test_dir = cfg.get("test_dir", "./test")
    test_output_dir = os.path.join(cfg.exp.dir, cfg.exp.name, "test_outputs")
    callbacks.append(TestFolderCallback(
        test_dir=test_dir,
        output_dir=test_output_dir,
        sr=_SR,
    ))

    uvr_export_dir = os.path.join(cfg.exp.dir, cfg.exp.name, "uvr_exports")
    callbacks.append(UVRExportCallback(uvr_export_dir, feature_dim=cfg.model.get("feature_dim", 256)))
    if cfg.get("early_stopping"):
        print_only(f"Instantiating early_stopping")
        callbacks.append(hydra.utils.instantiate(cfg.early_stopping))
    if cfg.get("checkpoint"):
        print_only(f"Instantiating checkpoint")
        checkpoint = hydra.utils.instantiate(cfg.checkpoint)
        callbacks.append(checkpoint)

    # Instantiate logger
    print_only(f"Instantiating logger <{cfg.logger._target_}>")
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "logs"), exist_ok=True)
    logger = hydra.utils.instantiate(cfg.logger)

    # Instantiate trainer — single GPU, no DDP
    print_only(f"Instantiating trainer")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Resume from last checkpoint if requested
    ckpt_path = None
    if cfg.get("resume", False):
        ckpt_dir = os.path.join(cfg.exp.dir, cfg.exp.name, "checkpoints")
        if os.path.isdir(ckpt_dir):
            # Find the most recently written real checkpoint — skip last.ckpt (symlink/alias)
            candidates = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.endswith(".ckpt") and f != "last.ckpt"
            ]
            if candidates:
                ckpt_path = max(candidates, key=os.path.getmtime)
                print_only(f"Resuming from checkpoint: {ckpt_path}")
            else:
                print_only(f"No checkpoints found in {ckpt_dir}, starting from scratch.")
        else:
            print_only(f"Checkpoint directory not found: {ckpt_dir}, starting from scratch.")

    trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    print_only("Training finished!")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(cfg.exp.dir, cfg.exp.name, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="configs/apollo.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Path to local weights file (.pth or .ckpt). If not set, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint in the experiment checkpoint directory.",
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        help="Path to folder containing LQ audio files to process each validation epoch. "
             "Outputs saved to <exp_dir>/test_outputs/epoch_NNNN/<filename>.wav",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    if args.weights_path:
        cfg.weights_path = args.weights_path
    if args.resume:
        cfg.resume = True
    if args.test_dir:
        cfg.test_dir = args.test_dir

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))

    train(cfg)