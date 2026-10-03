"""
degrade_audio.py -- Synthetic audio degradation pipeline for training data gen.

Replaces degrade_audio.bat. The codec/filter chain is defined in a JSON config
under utils/degrade/ instead of being hardcoded, so different degradation
recipes can be authored without editing code.

Launched from tui.py as a TUI screen, or standalone:
    python utils/degrade_audio.py --config utils/degrade/default.json --input "in.flac" --output "C:\\out"
    python utils/degrade_audio.py --config utils/degrade/default.json --input "C:\\in_folder" --output "C:\\out" --bulk

Chain step types:
    wma_encode   -- {bitrate}                          ffmpeg -c:a wmav2
    mp3_lame     -- {quality} or {bitrate}              ffmpeg -c:a libmp3lame
    mp3_fhg      -- {bitrate, enc_delay, codec_name}     acmenc (Fraunhofer IIS) --
                    requires "external_codecs.acmenc" set to a local acmenc.exe
                    path in the config; acmenc is a custom install, not shipped
    lowpass      -- {cutoff_hz, poles}                   ffmpeg -af lowpass
    highpass     -- {cutoff_hz, poles}                   ffmpeg -af highpass

Each step operates on the WAV output of the previous step. Steps that need a
compressed intermediate (mp3_lame, mp3_fhg, wma_encode) are followed
automatically by a decode-to-WAV pass before the next step -- the config only
lists the meaningful degradation passes, not the plumbing between them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aiff", ".aif"}

FHG_CODEC_NAME = "Fraunhofer IIS MPEG Layer-3 Codec (professional)"


class DegradeError(RuntimeError):
    pass


def _sanitize_basename(name: str) -> str:
    for ch in "()&":
        name = name.replace(ch, "")
    return name.replace(" ", "_")


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    cfg = json.loads(path.read_text())
    cfg.setdefault("tools", {})
    cfg["tools"].setdefault("ffmpeg", "ffmpeg")
    cfg.setdefault("external_codecs", {})
    if not cfg.get("chain") and "random" not in cfg:
        raise DegradeError(f"Config {path} has no 'chain' steps.")
    return cfg


def list_configs(configs_dir: Path) -> list[Path]:
    if not configs_dir.exists():
        return []
    return sorted(configs_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

def _run(cmd: list[str], print_fn) -> None:
    print_fn(f"    $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print_fn(result.stdout)
        print_fn(result.stderr)
        raise DegradeError(f"Command failed (exit {result.returncode}): {cmd[0]}")


def _step_wma_encode(step: dict, src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    bitrate = step.get("bitrate", "128k")
    out = tmpdir / f"gen{n}_wma.wma"
    _run([tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-c:a", "wmav2", "-b:a", str(bitrate), str(out)], print_fn)
    return out


def _step_mp3_lame(step: dict, src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    out = tmpdir / f"gen{n}_mp3.mp3"
    cmd = [tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-c:a", "libmp3lame"]
    if "quality" in step:
        cmd += ["-q:a", str(step["quality"])]
    elif "bitrate" in step:
        cmd += ["-b:a", str(step["bitrate"])]
    else:
        raise DegradeError("mp3_lame step needs 'quality' or 'bitrate'.")
    cmd.append(str(out))
    _run(cmd, print_fn)
    return out


def _step_mp3_fhg(step: dict, src_wav: Path, tmpdir: Path, n: int, external_codecs: dict, print_fn, outdir_override: Path | None = None) -> Path:
    bitrate = step.get("bitrate", 192)
    enc_delay = step.get("enc_delay", 672)
    codec_name = step.get("codec_name", FHG_CODEC_NAME)
    out = (outdir_override / f"gen{n}_mp3.mp3") if outdir_override else (tmpdir / f"gen{n}_mp3.mp3")
    acmenc = external_codecs.get("acmenc")
    if not acmenc:
        raise DegradeError(
            "mp3_fhg step requires 'external_codecs.acmenc' in the config -- "
            "set it to the path to acmenc.exe on this machine. acmenc is a "
            "custom local install and is not shipped with this repo."
        )
    if not shutil.which(acmenc) and not Path(acmenc).exists():
        raise DegradeError(f"acmenc not found at '{acmenc}' -- check external_codecs.acmenc in the config.")
    cmd = [
        acmenc, "-c", codec_name,
        "--enc-delay", str(enc_delay),
        f"-b{bitrate}", str(src_wav), str(out),
    ]
    _run(cmd, print_fn)
    return out


def _step_filter(step: dict, kind: str, src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    cutoff = step.get("cutoff_hz")
    if cutoff is None:
        raise DegradeError(f"{kind} step needs 'cutoff_hz'.")
    poles = step.get("poles", 2)
    out = tmpdir / f"gen{n}_{kind}.wav"
    af = f"{kind}=f={cutoff}:poles={poles}"
    _run([tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-af", af, "-c:a", "pcm_s16le", str(out)], print_fn)
    return out


def _decode_to_wav(src: Path, tmpdir: Path, n: int, tools: dict, print_fn) -> Path:
    out = tmpdir / f"gen{n}_dec.wav"
    _run([tools["ffmpeg"], "-y", "-i", str(src), "-vn", "-c:a", "pcm_s16le", str(out)], print_fn)
    return out


_COMPRESSED_STEPS = {"wma_encode", "mp3_lame", "mp3_fhg"}
_FILTER_STEPS = {"lowpass", "highpass"}


def run_chain(config: dict, input_path: str | Path, output_dir: str | Path, print_fn=print) -> Path:
    """Run the configured codec/filter chain on a single file. Returns the output path."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise DegradeError(f"Input not found: {input_path}")

    tools = config["tools"]
    external_codecs = config["external_codecs"]
    chain = config.get("chain")
    if not chain:
        raise DegradeError("This config is randomized: run it with --randomize.")
    basename = _sanitize_basename(input_path.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = output_dir / f"tmp_{basename}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    print_fn(f"Degradation pipeline starting")
    print_fn(f"  Input:  {input_path}")
    print_fn(f"  Output: {output_dir}")
    print_fn(f"  Config: {config.get('name', '(unnamed)')}  ({len(chain)} steps)")
    print_fn("")

    try:
        current = input_path
        for i, step in enumerate(chain, start=1):
            step_type = step.get("type")
            print_fn(f"[{i}/{len(chain)}] {step_type} ...")

            is_last = (i == len(chain))

            if step_type == "wma_encode":
                current = _step_wma_encode(step, current, tmpdir, i, tools, print_fn)
            elif step_type == "mp3_lame":
                current = _step_mp3_lame(step, current, tmpdir, i, tools, print_fn)
            elif step_type == "mp3_fhg":
                # mp3_fhg needs WAV input; decode first if the previous step left compressed audio.
                if current.suffix.lower() != ".wav":
                    current = _decode_to_wav(current, tmpdir, i, tools, print_fn)
                dest_dir = output_dir if is_last else None
                current = _step_mp3_fhg(step, current, tmpdir, i, external_codecs, print_fn, outdir_override=dest_dir)
                if is_last:
                    final = output_dir / f"{basename}_degraded.mp3"
                    if current != final:
                        shutil.move(str(current), str(final))
                    current = final
            elif step_type in _FILTER_STEPS:
                if current.suffix.lower() != ".wav":
                    current = _decode_to_wav(current, tmpdir, i, tools, print_fn)
                current = _step_filter(step, step_type, current, tmpdir, i, tools, print_fn)
            else:
                raise DegradeError(f"Unknown step type: {step_type!r}")

            print_fn("    Done.")

        # If the chain didn't end on mp3_fhg (which writes straight to output_dir),
        # copy/convert whatever we ended up with into the final output file.
        last_type = chain[-1].get("type")
        if last_type != "mp3_fhg":
            ext = ".mp3" if last_type in ("wma_encode", "mp3_lame") else ".wav"
            final = output_dir / f"{basename}_degraded{ext}"
            shutil.copy2(str(current), str(final))
            current = final

        print_fn("")
        print_fn(f"Complete. Output: {current}")
        return current
    finally:
        print_fn("Cleaning up temp files...")
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_bulk(config: dict, input_dir: str | Path, output_dir: str | Path, print_fn=print) -> list[Path]:
    input_dir = Path(input_dir)
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        print_fn(f"No audio files found in {input_dir}")
        return []

    print_fn(f"BULK MODE -- {len(files)} file(s) in {input_dir}")
    print_fn("")
    results = []
    for i, f in enumerate(files, start=1):
        print_fn(f"[BULK {i}/{len(files)}] {f.name}")
        try:
            results.append(run_chain(config, f, output_dir, print_fn=print_fn))
        except DegradeError as ex:
            print_fn(f"  ERROR: {ex} -- skipping.")
        print_fn("")
    print_fn(f"BULK MODE complete. {len(results)}/{len(files)} succeeded.")
    return results


# ---------------------------------------------------------------------------
# Randomized per-segment mode
# ---------------------------------------------------------------------------

def _wpick(rng: random.Random, weights: dict) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def sample_chain(rc: dict, rng: random.Random) -> list[dict]:
    """Draw one chain from the 'random' block of a recipe."""
    def mp3_pass() -> dict:
        if rng.random() < rc["mp3_vbr_prob"]:
            return {"type": "mp3_lame", "quality": int(_wpick(rng, rc["mp3_vbr_quality"]))}
        return {"type": "mp3_lame", "bitrate": int(_wpick(rng, rc["mp3_cbr_bitrate"]))}

    def wma() -> dict:
        return {"type": "wma_encode", "bitrate": _wpick(rng, rc["wma_bitrate"])}

    chain = [mp3_pass() for _ in range(int(_wpick(rng, rc["mp3_passes"])))]
    final = _wpick(rng, rc["final"])
    if final == "cbr192":
        chain.append({"type": "mp3_lame", "bitrate": 192})
    elif final == "cbr192_extra":
        chain.append({"type": "mp3_lame", "bitrate": 192})
        chain.append({"type": "mp3_lame", "bitrate": int(_wpick(rng, rc["final_extra_bitrate"]))})
    elif final == "vbr":
        chain.append({"type": "mp3_lame", "quality": int(_wpick(rng, rc["mp3_vbr_quality"]))})

    pos = _wpick(rng, rc["wma_position"])
    if pos == "first":
        chain.insert(0, wma())
    elif pos == "last":
        chain.append(wma())
    elif pos == "middle":
        chain.insert(rng.randint(1, len(chain) - 1), wma())
    elif pos == "double":
        chain.insert(0, wma())
        chain.append(wma())
    return chain


def describe_chain(chain: list[dict]) -> str:
    parts = []
    for s in chain:
        if s["type"] == "wma_encode":
            parts.append("wma" + str(s["bitrate"]).rstrip("k"))
        elif "quality" in s:
            parts.append(f"q{s['quality']}")
        else:
            parts.append(f"c{s['bitrate']}")
    return ">".join(parts)


def measure_offset(hq, lq, sr: int, max_lag: int) -> tuple[int, float]:
    """Integer-sample delay of lq relative to hq on the 0-4 kHz band, plus the correlation
    coefficient at that lag. lag > 0 means lq is late."""
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 4000, btype="low", fs=sr, output="sos")
    a = sosfiltfilt(sos, hq.mean(axis=1))
    b = sosfiltfilt(sos, lq.mean(axis=1))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    nfft = 1 << (2 * n - 1).bit_length()
    R = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)), nfft)
    lags = np.arange(-max_lag, max_lag + 1)
    r = R[lags % nfft]
    i = int(np.argmax(r))
    norm = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
    return int(lags[i]), float(r[i] / norm)


def _encode_chain(chain: list[dict], src_wav: Path, tmpdir: Path, tools: dict, sr: int, log) -> Path:
    current = src_wav
    for i, step in enumerate(chain, start=1):
        if step["type"] == "wma_encode":
            current = _step_wma_encode(step, current, tmpdir, i, tools, log)
        else:
            current = _step_mp3_lame(step, current, tmpdir, i, tools, log)
    out = tmpdir / "final.wav"
    _run([tools["ffmpeg"], "-y", "-i", str(current), "-vn", "-ar", str(sr), "-c:a", "pcm_s16le", str(out)], log)
    return out


def _segment_worker(job: dict) -> dict:
    import numpy as np
    import soundfile as sf

    res = {"name": job["name"], "chain": describe_chain(job["chain"]), "status": "ok"}
    tmpdir = Path(job["tmp_root"]) / job["name"]
    log: list[str] = []
    try:
        tmpdir.mkdir(parents=True, exist_ok=True)
        sr = job["sr"]
        hq, _ = sf.read(job["src"], start=job["rd_start"], stop=job["rd_stop"], always_2d=True, dtype="float64")
        core = hq[job["pad_start"]: job["pad_start"] + job["core_len"]]
        if len(core) == 0 or 20 * np.log10(np.sqrt(np.mean(core ** 2)) + 1e-12) < job["silence_dbfs"]:
            res["status"] = "rejected: silent"
            return res

        src_wav = tmpdir / "seg.wav"
        sf.write(str(src_wav), hq, sr, subtype="PCM_24")
        lq_wav = _encode_chain(job["chain"], src_wav, tmpdir, job["tools"], sr, log.append)
        lq, _ = sf.read(str(lq_wav), always_2d=True, dtype="float64")

        lag, corr = measure_offset(hq, lq, sr, job["max_lag"])
        res["lag"], res["corr"] = lag, round(corr, 4)
        if corr < job["min_corr"]:
            res["status"] = "rejected: weak alignment"
            return res

        if lag > 0:
            lq = lq[lag:]
        elif lag < 0:
            hq = hq[-lag:]
        start = job["pad_start"] + min(lag, 0)
        length = job["core_len"]
        if start < 0:
            length += start
            start = 0
        end = min(start + length, len(hq), len(lq))
        hq_c, lq_c = hq[start:end], lq[start:end]
        if len(hq_c) < job["min_len"]:
            res["status"] = "rejected: too short after alignment"
            return res

        tmp_hq, tmp_lq = job["out_hq"] + ".part", job["out_lq"] + ".part"
        sf.write(tmp_hq, hq_c, sr, subtype="PCM_24", format="WAV")
        sf.write(tmp_lq, lq_c, sr, subtype="PCM_24", format="WAV")
        os.replace(tmp_lq, job["out_lq"])
        os.replace(tmp_hq, job["out_hq"])
        res["seconds"] = round(len(hq_c) / sr, 3)
    except Exception as ex:
        res["status"] = f"error: {ex} {' | '.join(log[-4:])}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return res


def run_randomized(config: dict, input_path: str | Path, output_dir: str | Path,
                   seed: int = 0, workers: int | None = None, max_minutes: float | None = None,
                   print_fn=print) -> list[dict]:
    """Cut each source into segments, give every segment its own random chain, align it,
    and write LQ/HQ pairs to <output_dir>/LQ and <output_dir>/HQ."""
    import soundfile as sf

    rc = config.get("random")
    if not rc:
        raise DegradeError("Config has no 'random' block.")
    tools = config["tools"]
    input_path, output_dir = Path(input_path), Path(output_dir)
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise DegradeError(f"No audio files found in {input_path}")

    lq_dir, hq_dir, tmp_root = output_dir / "LQ", output_dir / "HQ", output_dir / "_tmp"
    lq_dir.mkdir(parents=True, exist_ok=True)
    hq_dir.mkdir(parents=True, exist_ok=True)

    seg_sec = float(rc.get("segment_sec", 30))
    pad_sec = float(rc.get("pad_sec", 1.0))
    min_sec = float(rc.get("min_segment_sec", 3.0))

    jobs: list[dict] = []
    skipped_existing = 0
    for f in files:
        try:
            info = sf.info(str(f))
        except Exception as ex:
            print_fn(f"  skip {f.name}: unreadable ({ex})")
            continue
        sr = info.samplerate
        if sr not in (32000, 44100, 48000):
            print_fn(f"  skip {f.name}: {sr} Hz is not an MP3 sample rate")
            continue
        seg_n, pad_n, total = int(seg_sec * sr), int(pad_sec * sr), info.frames
        base = _sanitize_basename(f.stem)
        for idx, s in enumerate(range(0, total, seg_n), start=1):
            e = min(s + seg_n, total)
            if (e - s) < min_sec * sr:
                continue
            name = f"{base}_s{idx:03d}"
            out_lq, out_hq = lq_dir / f"{name}.wav", hq_dir / f"{name}.wav"
            if out_lq.exists() and out_hq.exists():
                skipped_existing += 1
                continue
            rd_start, rd_stop = max(0, s - pad_n), min(total, e + pad_n)
            jobs.append({
                "name": name, "src": str(f), "sr": sr,
                "rd_start": rd_start, "rd_stop": rd_stop,
                "pad_start": s - rd_start, "core_len": e - s,
                "chain": sample_chain(rc, random.Random(f"{seed}:{name}")),
                "tools": tools, "tmp_root": str(tmp_root),
                "max_lag": int(rc.get("max_lag", 8192)),
                "min_corr": float(rc.get("min_corr", 0.5)),
                "silence_dbfs": float(rc.get("silence_dbfs", -60)),
                "min_len": int(min_sec * sr),
                "out_lq": str(out_lq), "out_hq": str(out_hq),
            })

    if max_minutes:
        rng = random.Random(seed)
        rng.shuffle(jobs)
        kept, total_min = [], 0.0
        for j in jobs:
            if total_min >= max_minutes:
                break
            kept.append(j)
            total_min += j["core_len"] / j["sr"] / 60
        jobs = sorted(kept, key=lambda j: j["name"])

    planned = sum(j["core_len"] / j["sr"] for j in jobs) / 60
    print_fn(f"Randomized degradation: {len(files)} file(s), {len(jobs)} segment(s), ~{planned:.1f} min"
             f" ({skipped_existing} already done)")
    if not jobs:
        return []

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    results: list[dict] = []
    done_sec, rejects = 0.0, {}
    with open(output_dir / "_chains.jsonl", "a", encoding="utf-8") as logf, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_segment_worker, j) for j in jobs]
        try:
            for n, fut in enumerate(as_completed(futures), start=1):
                r = fut.result()
                results.append(r)
                logf.write(json.dumps(r) + "\n")
                logf.flush()
                if r["status"] == "ok":
                    done_sec += r["seconds"]
                    print_fn(f"[{n}/{len(jobs)}] {r['name']}  lag={r['lag']}  corr={r['corr']}  {r['chain']}")
                else:
                    rejects[r["status"]] = rejects.get(r["status"], 0) + 1
                    print_fn(f"[{n}/{len(jobs)}] {r['name']}  {r['status']}  {r['chain']}")
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            print_fn("Interrupted. Rerun with the same settings to continue; finished pairs are kept.")
    shutil.rmtree(tmp_root, ignore_errors=True)

    print_fn(f"Kept {sum(1 for r in results if r['status'] == 'ok')}/{len(jobs)} segments, {done_sec / 60:.1f} min.")
    for reason, count in sorted(rejects.items()):
        print_fn(f"  {count} x {reason}")
    return results


# ---------------------------------------------------------------------------
# TUI entry point
# ---------------------------------------------------------------------------

def _screen_randomized(state: dict, console, _pick, _run_with_live_output, ROOT: Path, cfg_path: Path) -> None:
    """TUI flow for randomized recipes. Runs the CLI as a subprocess, like inference does."""
    from rich.markup import escape

    st = state.setdefault("degrade", {}).setdefault("random", {})
    input_dir = ROOT / "input"
    input_dir.mkdir(exist_ok=True)
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    all_label = (f"[ Process all {len(files)} file(s) in /input ]" if files
                 else "[ Process all in /input (folder empty) ]")
    items = [all_label, "[ Enter custom file/folder path ]"] + [f.name for f in files]

    idx = _pick("Randomized degrade -- select input", items, hint="Enter=select  Esc=back")
    if idx is None:
        return
    if idx == 0:
        if not files:
            console.print("[yellow]No files in /input.[/]")
            console.input("Press Enter to return.")
            return
        src = str(input_dir)
    elif idx == 1:
        console.clear()
        src = console.input("[cyan]Enter file or folder path:[/] ").strip().strip('"')
        if not src:
            return
    else:
        src = str(files[idx - 2])

    console.clear()
    console.print(f"[bold cyan]Randomized degrade[/]  --  recipe: {cfg_path.stem}\n")
    default_out = st.get("last_output") or str(ROOT / "output" / "pairs")
    out = console.input(f"[cyan]Output folder, LQ/ and HQ/ are created inside[/] (default: {escape(default_out)}): ").strip().strip('"') or default_out
    seed = console.input(f"[cyan]Seed[/] (default: {st.get('seed', 0)}): ").strip() or str(st.get("seed", 0))
    mins = console.input(f"[cyan]Max minutes of synthetic audio, 0 for no cap[/] (default: {st.get('max_minutes', 0)}): ").strip() or str(st.get("max_minutes", 0))
    try:
        seed_i, mins_f = int(seed), float(mins)
    except ValueError:
        console.print("[red]Seed must be a whole number and max minutes a number.[/]")
        console.input("Press Enter to return.")
        return

    st.update({"last_output": out, "seed": seed_i, "max_minutes": mins_f})
    cmd = [sys.executable, str(ROOT / "utils" / "degrade_audio.py"),
           "--config", str(cfg_path), "--input", src, "--output", out,
           "--randomize", "--seed", str(seed_i)]
    if mins_f > 0:
        cmd += ["--max_minutes", str(mins_f)]
    _run_with_live_output(cmd, f"Randomized degrade: {cfg_path.stem}")


def screen_degrade_audio(state: dict, console, _pick, _run_with_live_output, ROOT: Path) -> None:
    """Entry point called from tui.py."""
    configs_dir = ROOT / "utils" / "degrade"
    configs = list_configs(configs_dir)
    if not configs:
        console.clear()
        console.print(f"[red]No degrade configs found in {configs_dir}[/]")
        console.input("Press Enter to return.")
        return

    last_cfg = state.get("degrade", {}).get("last_config", "")
    start = next((i for i, c in enumerate(configs) if c.name == last_cfg), 0)
    items = [c.stem for c in configs]

    idx = _pick("Degrade Audio -- select config", items, hint="Enter=select  Esc=back", start=start)
    if idx is None:
        return
    cfg_path = configs[idx]
    state.setdefault("degrade", {})["last_config"] = cfg_path.name

    try:
        is_random = "random" in load_config(cfg_path)
    except DegradeError as ex:
        console.print(f"[red]Error: {ex}[/]")
        console.input("Press Enter to return.")
        return
    if is_random:
        _screen_randomized(state, console, _pick, _run_with_live_output, ROOT, cfg_path)
        return

    input_dir = ROOT / "input"
    output_dir = ROOT / "output"
    input_dir.mkdir(exist_ok=True)

    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    bulk_label = f"[ Process all {len(files)} file(s) in /input ]" if files else "[ Process all in /input (folder empty) ]"
    items = [bulk_label, "[ Enter custom file/folder path ]"] + [f.name for f in files]

    idx2 = _pick("Degrade Audio -- select input", items, hint="Enter=select  Esc=back")
    if idx2 is None:
        return

    console.clear()
    console.print(f"[bold cyan]Degrade Audio[/]  --  config: {cfg_path.stem}\n")

    lines = []
    def _print(*args):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        console.print(msg)

    try:
        config = load_config(cfg_path)
        if idx2 == 0:
            if not files:
                console.print("[yellow]No files in /input.[/]")
            else:
                run_bulk(config, input_dir, output_dir, print_fn=_print)
        elif idx2 == 1:
            path = console.input("[cyan]Enter file or folder path:[/] ").strip().strip('"')
            if not path:
                return
            p = Path(path)
            if p.is_dir():
                run_bulk(config, p, output_dir, print_fn=_print)
            else:
                run_chain(config, p, output_dir, print_fn=_print)
        else:
            run_chain(config, files[idx2 - 2], output_dir, print_fn=_print)
    except DegradeError as ex:
        console.print(f"[red]Error: {ex}[/]")
    except Exception as ex:
        console.print(f"[red]Unexpected error: {ex}[/]")

    console.input("\n[dim]Press Enter to return to menu[/]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic audio degradation pipeline.")
    parser.add_argument("--config", required=True, help="Path to a degrade config JSON.")
    parser.add_argument("--input", required=True, help="Input audio file or folder (with --bulk).")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--bulk", action="store_true", help="Treat --input as a folder and process every audio file in it.")
    parser.add_argument("--randomize", action="store_true", help="Segment each file and give every segment its own random chain; writes <output>/LQ and <output>/HQ.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --randomize.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel segments for --randomize (default: CPU count - 1).")
    parser.add_argument("--max_minutes", type=float, default=None, help="Cap the synthetic total for --randomize.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.randomize:
        run_randomized(config, args.input, args.output, seed=args.seed, workers=args.workers, max_minutes=args.max_minutes)
    elif args.bulk:
        run_bulk(config, args.input, args.output)
    else:
        run_chain(config, args.input, args.output)


if __name__ == "__main__":
    main()
