# rebuild_adaptive_v1041.py
#
# Rebuild the adaptive checkpoint from the stable v9.4 baseline using all
# trusted chat teaching pairs plus stability replay.
#
# The existing online checkpoint and learning state are backed up before
# replacement.  The canonical v9.4 baseline is never overwritten.

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


BASE_MODEL = Path("model/model-llm-try-nagato-chat-v94.pt")
ONLINE_MODEL = Path("model/model-gpu-v1.6.2-online.pt")
CHAT_LOG = Path("data/chat_history.jsonl")
STATE = Path("data/chat_learning_state.json")
TEMP_STATE = Path("data/chat_learning_state_v1041_rebuild.json")
TOKENIZER = Path("model/tokenizer-v0.7-bpe.json")
STABILITY = Path("data/stability_replay_v104.jsonl")


def main() -> None:
    print("=" * 88)
    print(" LLM_TRY v10.4.1 Adaptive Checkpoint Rebuild")
    print("=" * 88)

    required = [BASE_MODEL, CHAT_LOG, TOKENIZER, STABILITY]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = Path("backup") / f"v1041-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    if ONLINE_MODEL.exists():
        shutil.copy2(ONLINE_MODEL, backup_dir / ONLINE_MODEL.name)
        print("Backed up model :", backup_dir / ONLINE_MODEL.name)

    if STATE.exists():
        shutil.copy2(STATE, backup_dir / STATE.name)
        print("Backed up state :", backup_dir / STATE.name)

    TEMP_STATE.parent.mkdir(parents=True, exist_ok=True)
    TEMP_STATE.write_text(
        json.dumps(
            {"version": "v1.6.2", "trained_fingerprints": []},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "online_train.py",
        "--chat-data", str(CHAT_LOG),
        "--base-model", str(BASE_MODEL),
        "--tokenizer", str(TOKENIZER),
        "--output", str(ONLINE_MODEL),
        "--state", str(TEMP_STATE),
        "--stability-data", str(STABILITY),
        "--stability-weight", "1",
        "--learning-rate", "1e-5",
        "--trusted-replay-weight", "1",
        "--manual-weight", "4",
    ]

    print()
    print("Rebuilding from stable baseline...")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"Rebuild failed: exit={completed.returncode}")
        print("Original state backup is preserved at:", backup_dir)
        raise SystemExit(completed.returncode)

    if not ONLINE_MODEL.exists() or not TEMP_STATE.exists():
        raise RuntimeError("Rebuild completed without expected output files.")

    shutil.copy2(TEMP_STATE, STATE)
    TEMP_STATE.unlink(missing_ok=True)

    print()
    print("Rebuild completed.")
    print("Adaptive model :", ONLINE_MODEL)
    print("Learning state :", STATE)
    print("Backup         :", backup_dir)
    print()
    print("Next:")
    print("  python eval_adaptive_preservation_v103.py")
    print("  python eval_multiconcept_incremental_v104.py")


if __name__ == "__main__":
    main()
