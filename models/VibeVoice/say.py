import os, sys, argparse, subprocess, shlex, pathlib

def main():
    ap = argparse.ArgumentParser(description="VibeVoice runner for single or multi-speaker scripts")
    ap.add_argument("--model_path", default="microsoft/VibeVoice-1.5B")
    # Choose EITHER a script file OR inline text
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--script_path", help="Path to a multi-speaker script (e.g., 'Speaker 1:' / 'Speaker 2:')")
    group.add_argument("--text", help="Inline text (single speaker)")
    # One or more speakers, in order, mapped to 'Speaker 1:', 'Speaker 2:', ...
    ap.add_argument("--speakers", nargs="+", default=["en-Alice_woman"],
                    help="Speaker names to map in order to 'Speaker 1:', 'Speaker 2:', etc.")
    ap.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    ap.add_argument("--out", default="outputs/out.wav")
    args = ap.parse_args()

    # Ensure folders exist
    pathlib.Path("demo/text_examples").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)

    # If inline text is provided, write it to a temp file; otherwise use the provided script_path
    if args.script_path:
        txt_path = args.script_path
    else:
        txt_path = "demo/text_examples/one_line.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(args.text)

    # Build the --speaker_names part
    speakers_cmd = " ".join(shlex.quote(s) for s in args.speakers)

    # Call the official file runner
    cmd = (
        f"{shlex.quote(sys.executable)} demo/inference_from_file.py "
        f"--model_path {shlex.quote(args.model_path)} "
        f"--txt_path {shlex.quote(txt_path)} "
        f"--speaker_names {speakers_cmd} "
        f"--device {shlex.quote(args.device)}"
    )
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

    # Grab the newest WAV from outputs and copy to --out if needed
    outs = sorted(pathlib.Path('outputs').glob('*.wav'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outs:
        sys.exit('No WAV produced')
    last = outs[0]
    if args.out != str(last):
        pathlib.Path(args.out).write_bytes(last.read_bytes())
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
