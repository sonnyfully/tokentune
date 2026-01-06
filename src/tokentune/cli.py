from future import annotations

import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

from tokentune.audio.io import load_audio, to_mono, resample, peak_normalize, rms
from tokentune.audio.slice import slice_audio
from tokentune.data.manifest import write_manifest

def preprocess_data(input_dir: Path | str,
                    output_dir: Path | str,
                    config: str) -> None:
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    
    target_sr = int(cfg["target_sr"])
    clip_seconds = float(cfg["clip_seconds"])
    hop_seconds = float(cfg["hop_seconds"])
    silence_rms_threshold = float(cfg.get("silence_rms_threshold", 0.0))
    peak_target = float(cfg.get("peak_target", 0.95))

    manifest_path = output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

wav_paths = sorted(list(input_dir.glob("*.wav")))

if not wav_paths:
    raise ValueError(f"No WAV files found in {input_dir}")

clip_count = 0

for wav_path in tqdm(wav_paths, desc="Preprocessing audio files"):
    waveform, sample_rate = load_audio(wav_path)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sample_rate, target_sr)
    waveform = peak_normalize(waveform, peak_target=peak_target)
    
    clips = slice_audio(waveform, sample_rate, clip_seconds, hop_seconds)
    for i, clip in enumerate(clips):
        clip_rms = rms(clip)
        if clip_rms < silence_rms_threshold:
            continue

        out_name = f"{wav_path.stem}_clip{i:05d}.wav"
        out_path = output_dir / out_name

        from tokentune.audio.io import save_wav
        save_wav(clip, sample_rate, out_path)

        record = {
            "clip_path": str(out_path.as_posix()),
            "source_path": str(wav_path.as_posix()),
            "sample_rate": sample_rate,
            "clip_seconds": clip_seconds,
            "hop_seconds": hop_seconds,

            "clip_rms": clip_rms,
        }
        write_manifest(manifest_path, record)
        clip_count += 1

    print(f"Preprocessed {clip_count} clips to {output_dir}")
    print(f"Manifest written to {manifest_path}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(tokentune)
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Preprocess audio files and create a manifest")
    pp.add_argument("--input_dir", type=str, required=True)
    pp.add_argument("--output_dir", type=str, required=True)
    pp.add_argument("--config", type=str, required=True)
    
    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "preprocess":
        preprocess_data(args.input_dir, args.output_dir, args.config)
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()