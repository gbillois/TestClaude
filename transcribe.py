#!/usr/bin/env python3
"""
Transcription locale avec faster-whisper
Usage : python transcribe.py <fichier> [options]
"""

import argparse
import os
import sys
import time
from pathlib import Path


MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".flv", ".wmv", ".m4v"}


def format_timestamp(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")


def write_txt(segments, output_path: Path, show_timestamps: bool):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            if show_timestamps:
                f.write(f"[{format_timestamp(seg.start)}] {seg.text.strip()}\n")
            else:
                f.write(seg.text.strip() + "\n")


def transcribe(args):
    from faster_whisper import WhisperModel

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERREUR] Fichier introuvable : {input_path}")
        sys.exit(1)

    ext = input_path.suffix.lower()
    if ext not in AUDIO_EXTS | VIDEO_EXTS:
        print(f"[ERREUR] Format non supporté : {ext}")
        print(f"  Audio : {', '.join(sorted(AUDIO_EXTS))}")
        print(f"  Vidéo : {', '.join(sorted(VIDEO_EXTS))}")
        sys.exit(1)

    # Chargement du modèle
    print(f"\n  Modèle      : {args.model}")
    print(f"  Fichier     : {input_path.name}")
    print(f"  Langue      : {args.language or 'auto-détection'}")
    print(f"  Device      : {args.device}")
    print()

    print("Chargement du modèle (téléchargement automatique si absent)...")
    t0 = time.time()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type="int8",
        download_root=None,  # cache HuggingFace par défaut
    )
    print(f"Modèle chargé en {time.time() - t0:.1f}s\n")

    # Transcription
    print("Transcription en cours...")
    t1 = time.time()
    segments_gen, info = model.transcribe(
        str(input_path),
        language=args.language or None,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    # Matérialisation + affichage en temps réel
    segments = []
    for seg in segments_gen:
        segments.append(seg)
        ts = format_timestamp(seg.start)
        print(f"  [{ts}] {seg.text.strip()}")

    elapsed = time.time() - t1
    duration = info.duration
    rtf = elapsed / duration if duration else 0

    print(f"\nTerminé en {elapsed:.1f}s  (durée audio : {duration:.1f}s, RTF : {rtf:.2f}x)")
    print(f"Langue détectée : {info.language}  (confiance : {info.language_probability:.0%})\n")

    # Écriture des fichiers de sortie
    stem = input_path.stem
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    if args.format in ("txt", "all"):
        out = out_dir / f"{stem}.txt"
        write_txt(segments, out, show_timestamps=not args.no_timestamps)
        print(f"TXT  -> {out}")

    if args.format in ("srt", "all"):
        out = out_dir / f"{stem}.srt"
        write_srt(segments, out)
        print(f"SRT  -> {out}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Transcription locale audio/vidéo avec faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python transcribe.py video.mp4
  python transcribe.py audio.mp3 --language fr --model large-v3
  python transcribe.py interview.wav --format srt --model medium
  python transcribe.py podcast.m4a --format all --output-dir ./sorties
        """,
    )

    parser.add_argument("input", help="Fichier audio ou vidéo à transcrire")
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="medium",
        help="Taille du modèle Whisper (défaut : medium). Plus grand = plus précis mais plus lent.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Langue forcée (ex: fr, en, es). Auto-détection si absent.",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "all"],
        default="txt",
        help="Format de sortie (défaut : txt)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device de calcul (défaut : auto)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Dossier de sortie (défaut : même dossier que le fichier source)",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Ne pas inclure les timestamps dans le fichier TXT",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n--- Modèles disponibles ---")
        print(f"  {'Modèle':<12} {'Taille':<10} {'VRAM':<10} Vitesse relative")
        print(f"  {'tiny':<12} {'~75 Mo':<10} {'~1 Go':<10} ~32x")
        print(f"  {'base':<12} {'~145 Mo':<10} {'~1 Go':<10} ~16x")
        print(f"  {'small':<12} {'~466 Mo':<10} {'~2 Go':<10} ~6x")
        print(f"  {'medium':<12} {'~1.5 Go':<10} {'~5 Go':<10} ~2x  <- défaut")
        print(f"  {'large-v2':<12} {'~3 Go':<10} {'~10 Go':<10} ~1x")
        print(f"  {'large-v3':<12} {'~3 Go':<10} {'~10 Go':<10} ~1x  (meilleure précision)")
        sys.exit(0)

    args = parser.parse_args()

    # Résolution du device
    if args.device == "auto":
        try:
            import ctranslate2
            args.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            args.device = "cpu"

    print("=" * 50)
    print("  faster-whisper — transcription locale")
    print("=" * 50)

    transcribe(args)


if __name__ == "__main__":
    main()
