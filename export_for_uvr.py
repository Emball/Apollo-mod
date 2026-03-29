"""
export_for_uvr.py

Converts a PyTorch Lightning training checkpoint (.ckpt) into the format
UVR's apollo_inference.py expects via from_pretrain().

Usage:
    # Base model (feature_dim=256)
    python export_for_uvr.py --ckpt ./Exps/Apollo/checkpoints/last.ckpt --out ./my_apollo_uvr.ckpt

    # Universal model (feature_dim=384)
    python export_for_uvr.py --ckpt ./Exps/Apollo_Universal/checkpoints/last.ckpt --out ./my_apollo_uni_uvr.ckpt --feature_dim 384

    # Fix an already-exported file that has hann_win in it
    python export_for_uvr.py --fix ./my_apollo_uvr.ckpt --out ./my_apollo_uvr_fixed.ckpt
"""

import argparse
import torch


# Keys that exist in our modified apollo.py but not in UVR's version
# These must be stripped before UVR can load the checkpoint
_KEYS_TO_STRIP = {"hann_win"}


def convert(ckpt_path: str, out_path: str, feature_dim: int = 256):
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    raw_state = ckpt["state_dict"]

    # Handle both Lightning format (audio_model. prefix) and bare state dicts
    if any(k.startswith("audio_model.") for k in raw_state.keys()):
        model_state = {
            k.replace("audio_model.", ""): v
            for k, v in raw_state.items()
            if k.startswith("audio_model.")
        }
        print("Detected Lightning checkpoint format (audio_model. prefix)")
    else:
        model_state = dict(raw_state)
        print("Detected bare state dict format (no prefix)")

    if not model_state:
        raise ValueError(
            "No model keys found in checkpoint. "
            "Check that this is a valid Apollo training checkpoint."
        )

    # Strip keys that UVR's apollo.py doesn't have
    stripped = []
    for key in _KEYS_TO_STRIP:
        if key in model_state:
            model_state.pop(key)
            stripped.append(key)
    if stripped:
        print(f"Stripped UVR-incompatible keys: {stripped}")

    uvr_dict = {
        "model_name": "Apollo",
        "state_dict": model_state,
        "model_args": {
            "sr": 44100,
            "win": 20,
            "feature_dim": feature_dim,
            "layer": 6,
        },
        "infos": {
            "training_epoch": ckpt.get("epoch", "unknown"),
            "val_loss": "see checkpoint filename",
        },
    }

    torch.save(uvr_dict, out_path)
    print(f"Saved UVR-compatible model to: {out_path}")
    print(f"  Keys in state_dict: {len(model_state)}")
    print(f"  feature_dim: {feature_dim}")
    print(f"  Epoch: {ckpt.get('epoch', 'unknown')}")


def fix(ckpt_path: str, out_path: str):
    """Strip UVR-incompatible keys from an already-exported UVR checkpoint."""
    print(f"Loading existing UVR export: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise ValueError("File doesn't look like a UVR export — no 'state_dict' key found.")

    stripped = []
    for key in _KEYS_TO_STRIP:
        if key in ckpt["state_dict"]:
            ckpt["state_dict"].pop(key)
            stripped.append(key)

    if stripped:
        print(f"Stripped: {stripped}")
        # Make sure infos key exists
        if "infos" not in ckpt:
            ckpt["infos"] = {}
        torch.save(ckpt, out_path)
        print(f"Fixed checkpoint saved to: {out_path}")
    else:
        print("No incompatible keys found — file may already be compatible.")
        torch.save(ckpt, out_path)
        print(f"Saved unchanged to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Path to Lightning checkpoint (.ckpt) from training to convert",
    )
    parser.add_argument(
        "--fix",
        default=None,
        help="Path to an already-exported UVR checkpoint that needs incompatible keys stripped",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for UVR-compatible file",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=256,
        help="feature_dim used during training (256 for base, 384 for universal)",
    )
    args = parser.parse_args()

    if args.fix:
        fix(args.fix, args.out)
    elif args.ckpt:
        convert(args.ckpt, args.out, args.feature_dim)
    else:
        parser.error("Provide either --ckpt (convert from Lightning) or --fix (strip bad keys from existing export)")
