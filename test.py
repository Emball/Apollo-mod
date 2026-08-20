import torch
import torchaudio
import json
import os
from omegaconf import OmegaConf
import argparse
import look2hear.models
import look2hear.models.apollo
from look2hear.metrics import MetricsTracker
from look2hear.utils import print_only
import warnings
warnings.filterwarnings("ignore")

class Owndata():
    def __init__(self, root):
        self.root = root
        self.data_lists = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file == "codec_wav.wav":
                    self.data_lists.append(os.path.join(root, file))

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, idx):
        ori  = self.data_lists[idx].replace("codec_wav.wav", "ori_wav.wav")
        ori_audio   = torchaudio.load(ori)[0]
        codec_audio = torchaudio.load(self.data_lists[idx])[0]
        return ori_audio, codec_audio, ori


def test(cfg):
    data_val = Owndata("./codec-test")

    cfg.model.pop("_target_", None)
    model = look2hear.models.apollo.Apollo(
        sr=cfg.model.sr,
        win=cfg.model.win,
        feature_dim=cfg.model.feature_dim,
        layer=cfg.model.layer,
    )
    ckpt_path = os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()

    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/"), exist_ok=True)
    metrics = MetricsTracker(
        save_file=os.path.join(cfg.exp.dir, cfg.exp.name, "results/") + "metrics.csv"
    )
    length = len(data_val)
    for idx in range(length):
        ori_wav, codec_wav, key = data_val[idx]
        ori_wav   = ori_wav.cuda()
        codec_wav = codec_wav.unsqueeze(0).cuda()
        with torch.no_grad():
            ests = model(codec_wav)
            torchaudio.save(
                key.replace("ori_wav.wav", "ests_wav.wav"),
                ests.squeeze(0).cpu(), 44100,
            )
            metrics(ori_wav, ests, key)

        if idx % 10 == 0:
            dicts = metrics.update()
            print_only(
                f"Processed {idx}/{length} -- "
                f"SDR: {dicts['sdr']}, SI-SNR: {dicts['si-snr']}, VISQOL: {dicts['visqol']}"
            )

    metrics.final()
    print_only("Finished testing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="runs/Apollo/config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    test(cfg)
