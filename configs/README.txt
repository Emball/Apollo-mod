APOLLO CONFIG REFERENCE
=======================

Two configs are provided:
  apollo.yaml      — base model (feature_dim=256), recommended for most finetuning
  apollo_uni.yaml  — universal model (feature_dim=384), larger, needs more VRAM


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
exp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dir       Root folder for all experiment outputs (checkpoints, logs, val audio).
name      Subfolder name for this run. Change this to avoid overwriting a previous run.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
optimizations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
tf32                Enable TF32 matmuls on Ampere+ GPUs (3000/4000 series). Negligible
                    quality loss, meaningful speed boost. Leave true unless you have a
                    reason to need full float32 precision.

cudnn_benchmark     Let cuDNN benchmark conv algorithms on the first batch and pick the
                    fastest one. Best left true for fixed input shapes (which Apollo has).
                    Turn off if your input sizes vary wildly between runs.

expandable_segments Reduces CUDA allocator fragmentation, which helps avoid surprise OOMs
                    mid-run. Leave true.

triton_cache        Cache compiled Triton kernels to disk so they don't recompile every
                    run. Cache lives in .triton_cache/. Leave true.



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
n_layers_to_freeze  Number of BSNet layers to freeze (plus the BN front-end always).
                    Higher = less VRAM, faster steps, less risk of catastrophic forgetting,
                    but less flexibility for the model to adapt.
                    Recommended: 4 for base model on 11GB VRAM, 3 for universal.

hf_boost            Extra loss weight applied to high frequencies. 1.0 = flat (no boost).
                    Values around 1.5 push the model to pay more attention to treble detail.
                    Too high and it will over-sharpen. Don't go above 2.0.

val_save_interval   Save rendered validation audio to disk every N epochs. The same fixed
                    set of val chunks is used every time so you can track improvement by ear.
                    Loss is still computed every epoch regardless.

val_audio_pairs     How many val chunks to render to disk per save interval. These are
                    selected on the first val run and locked in — diverse across songs.

grad_accum_steps    Accumulate gradients over N batches before stepping the optimizer.
                    Simulates a larger effective batch size without extra VRAM.
                    grad_accum_steps: 2 with batch_size: 2 = effective batch size of 4.
                    Leave at 1 if you're not VRAM constrained.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
datas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_dir           Where chunked training pairs are stored. Auto-generated from data/train/
                    on first run. Delete this folder to force re-chunking (e.g. after changing
                    segment_sec).

eval_dir            Same as above for validation data.

sr                  Sample rate. Apollo expects 44100. Don't change this.

segment_sec         Length of audio chunks in seconds. Longer = more context for the model
                    to learn from, but more VRAM per chunk.
                    Changing this requires deleting chunks/ and re-chunking.

batch_size          Number of chunks per training step. Higher = more stable gradients but
                    more VRAM. With segment_sec: 4 on an 11GB card, batch_size: 2 is about
                    the limit.

num_workers         Number of CPU workers prefetching data. More workers = less GPU idle
                    time between batches. Set to number of free CPU cores, max around 8.

pin_memory          Allocate DataLoader batches in page-locked RAM for faster CPU→GPU
                    transfers. Uses idle system RAM. Leave true unless you're RAM constrained.

val_bootstrap_chunks  If no data/val/ folder exists, this many chunks are copied from the
                    training set to use as validation. Picked round-robin across songs.

augmentation        Two categories: live (applied every epoch during training) and
                    cached (baked into chunk files on disk at chunking time).
                    All augmentations are always applied identically to both LQ and HQ
                    so the model never sees a mismatch between input and target.

  augmentation.live     Applied by the dataloader each epoch. Use for cheap ops.
  augmentation.cached   Baked into chunk WAV files once at chunking time. Use for
                        expensive ops like pitch_shift and mp3_degradation so the
                        cost is paid once, not every epoch.

  cached.cached_variants    Number of augmented variants to write per chunk in addition
                            to the clean original. e.g. variants: 2 means each chunk
                            produces 3 files: original + 2 augmented copies.
                            Defaults to 1.

  Each augmentation has the same options in both live and cached blocks:
  Differences: live uses prob (per-sample probability), cached uses fraction
  (fraction of chunks that receive the augmentation across the dataset).

  mono_channel.prob/fraction   Picks L or R randomly per chunk instead of feeding
                               both near-identical stereo channels as unique data.

  pitch_shift.prob/fraction    Transparent pitch shift via torch-pitch-shift.
  pitch_shift.semitones_max    Max shift in either direction in semitones.

  noise.prob/fraction          Matched Gaussian noise added to both LQ and HQ.
  noise.sigma                  Noise standard deviation. 0.002 is very subtle.

  mp3_degradation.prob/fraction  Random CBR MP3 re-encode pass on LQ only.
  mp3_degradation.kbps_min       Minimum bitrate.
  mp3_degradation.kbps_max       Maximum bitrate.

  gain.prob/fraction    Random gain shift. Output clamped to [-1, 1].
  gain.db_max           Max shift in dB (applied as ±db_max).
  polarity.prob/fraction  Polarity flip (multiply by -1). Essentially free.

  NOTE: Chunks are always saved as 16-bit PCM WAV regardless of source bit depth.
        Only WAV files are accepted in data/ — MP3/FLAC/etc will be rejected.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
feature_dim         Width of the model. 256 = base, 384 = universal. Must match the
                    pretrained weights you're loading.
layer               Number of BSNet layers. Always 6 for pretrained Apollo weights.
win                 STFT window size in ms. Always 20 for pretrained Apollo weights.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
type                Which optimizer to use:
                      adamw        Standard 32-bit AdamW. Default. No extra deps.
                      adamw_8bit   8-bit AdamW via bitsandbytes. Cuts optimizer state
                                   VRAM by ~75% with no meaningful quality difference.
                                   Requires: pip install bitsandbytes
                      cpu_offload  32-bit AdamW with momentum states stored in CPU RAM
                                   instead of VRAM. Saves ~200-400 MB VRAM. Small
                                   slowdown on the optimizer step.

lr_g                Learning rate for the generator (Apollo model).
lr_d                Learning rate for the discriminator.
weight_decay        Weight decay applied to both optimizers.
betas_g             Adam beta1, beta2 for the generator optimizer.
betas_d             Adam beta1, beta2 for the discriminator optimizer. Lower beta1 (0.5)
                    is standard practice for GAN discriminators.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
scheduler_g / scheduler_d
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
step_size           Decay the learning rate every N epochs.
gamma               Multiply LR by this value each step. 0.98 = 2% reduction per step.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
system
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gradient_checkpointing  Recompute activations during backward instead of storing them.
                        Saves 30-40% VRAM at the cost of ~30% more compute per step.
                        Useful if you want to push segment_sec or batch_size higher
                        and are hitting OOM.

grad_accum_steps    Mirrors training.grad_accum_steps. Don't change this directly —
                    change it under training instead.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
early_stopping
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
patience            Stop training if val_loss doesn't improve for this many val checks.
                    In apollo_uni.yaml this is set very high (2000) to effectively disable
                    it — lower it if you want early stopping to actually trigger.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
checkpoint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
save_top_k          How many checkpoints to keep. -1 = keep all. Set to e.g. 5 to only
                    keep the 5 best by val_loss and save disk space.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
trainer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
devices             Which GPU index to use. [0] = first GPU.
max_epochs          Hard cap on training epochs.
precision           16-mixed = fp16 mixed precision. Recommended. Use 32 for debugging.
fast_dev_run        Set true to run 1 train + 1 val batch then exit. Good for sanity
                    checking a new setup without waiting.

val_check_interval  (apollo_uni only) Run validation every N steps instead of every epoch.
                    Useful when epochs are very long.
