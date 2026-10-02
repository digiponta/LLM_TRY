# sweep_rebuild_v1042.py
#
# LLM_TRY v10.4.2 balanced adaptive rebuild sweep.
#
# Trains candidate adaptive checkpoints from the stable v9.4 baseline with
# multiple preservation/learning strengths, evaluates each candidate, and
# installs the best passing candidate. Existing online model/state are backed up.

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
import shutil
import subprocess
import sys
import time


BASE_MODEL = Path("model/model-llm-try-nagato-chat-v94.pt")
ONLINE_MODEL = Path("model/model-gpu-v1.6.2-online.pt")
CHAT_LOG = Path("data/chat_history.jsonl")
STATE = Path("data/chat_learning_state.json")
TOKENIZER = Path("model/tokenizer-v0.7-bpe.json")
STABILITY = Path("data/stability_replay_v104.jsonl")
WORK = Path("model/v1042_sweep")

CONFIGS = [
    {"manual_weight": 4, "stability_weight": 1, "lr": "1e-5"},
    {"manual_weight": 6, "stability_weight": 1, "lr": "1e-5"},
    {"manual_weight": 8, "stability_weight": 1, "lr": "1e-5"},
    {"manual_weight": 6, "stability_weight": 2, "lr": "1e-5"},
    {"manual_weight": 8, "stability_weight": 2, "lr": "1e-5"},
    {"manual_weight": 6, "stability_weight": 1, "lr": "1.5e-5"},
]

BASELINE = [
    "あなたは誰ですか",
    "AIとは",
    "LLMとは",
    "CUDAとは",
    "量子力学とは",
]

LEARNED = [
    "宇宙とは",
    "数学とは",
    "文学とは",
]


def normalize_text(text: str) -> str:
    return "".join(text.strip().split())


def load_expected_learned(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    if not path.exists():
        return expected
    targets = set(LEARNED)
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        source = str(row.get("source", ""))
        if user in targets and answer and source in (
            "chat-manual", "chat-approved", "chat-recovery"
        ):
            expected[user] = answer
    return expected


def answer_fidelity(actual: str, expected: str) -> float:
    a = normalize_text(actual)
    e = normalize_text(expected)
    if not a or not e:
        return 0.0
    return SequenceMatcher(None, a, e).ratio()

UNKNOWN = [
    "ブラックホールとは",
    "相対性理論とは",
    "化学とは",
]


def extract_answer(text: str) -> str:
    for line in text.splitlines():
        pos = line.find("AI> ")
        if pos >= 0:
            return line[pos + 4:].strip()
    return ""


def extract_gate(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("[gate="):
            return line.strip()
    return ""


def run_chat(model: Path, prompt: str) -> tuple[str, str]:
    proc = subprocess.run(
        [
            sys.executable, "-X", "utf8", "chat.py",
            "--model", str(model),
            "--temperature", "0",
        ],
        input=prompt + "\nexit\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"chat.py failed for {prompt!r}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return extract_answer(proc.stdout), extract_gate(proc.stdout)


def write_empty_state(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"version": "v1.6.2", "trained_fingerprints": []},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def evaluate(model: Path, expected_learned: dict[str, str]) -> dict:
    baseline_ok = 0
    learned_ok = 0
    unknown_ok = 0
    details: list[str] = []

    for q in BASELINE:
        ans, gate = run_chat(model, q)
        ok = bool(ans) and ans != "未学習です" and "gate=KNOWN" in gate
        baseline_ok += int(ok)
        details.append(f"BASE {'PASS' if ok else 'FAIL'} {q} -> {ans}")

    learned_fidelities: list[float] = []
    for q in LEARNED:
        ans, gate = run_chat(model, q)
        ok = bool(ans) and ans != "未学習です" and "gate=KNOWN" in gate
        learned_ok += int(ok)
        expected = expected_learned.get(q, "")
        fidelity = answer_fidelity(ans, expected) if expected else (1.0 if ok else 0.0)
        learned_fidelities.append(fidelity)
        details.append(
            f"LEARN {'PASS' if ok else 'FAIL'} {q} -> {ans} "
            f"[fidelity={fidelity:.3f}]"
        )

    for q in UNKNOWN:
        ans, gate = run_chat(model, q)
        ok = ans == "未学習です"
        unknown_ok += int(ok)
        details.append(f"UNKN {'PASS' if ok else 'FAIL'} {q} -> {ans}")

    total = baseline_ok + learned_ok + unknown_ok
    strict_pass = (
        baseline_ok == len(BASELINE)
        and learned_ok == len(LEARNED)
        and unknown_ok == len(UNKNOWN)
    )
    return {
        "baseline": baseline_ok,
        "learned": learned_ok,
        "unknown": unknown_ok,
        "total": total,
        "strict_pass": strict_pass,
        "learned_fidelity": (
            sum(learned_fidelities) / len(learned_fidelities)
            if learned_fidelities else 0.0
        ),
        "details": details,
    }


def main() -> None:
    print("=" * 96)
    print(" LLM_TRY v10.4.3 Quality-Aware Adaptive Rebuild Sweep")
    print("=" * 96)

    required = [BASE_MODEL, CHAT_LOG, TOKENIZER, STABILITY]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = Path("backup") / f"v1042-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if ONLINE_MODEL.exists():
        shutil.copy2(ONLINE_MODEL, backup_dir / ONLINE_MODEL.name)
    if STATE.exists():
        shutil.copy2(STATE, backup_dir / STATE.name)

    WORK.mkdir(parents=True, exist_ok=True)
    expected_learned = load_expected_learned(CHAT_LOG)
    missing_expected = [q for q in LEARNED if q not in expected_learned]
    if missing_expected:
        raise RuntimeError(
            "Missing trusted teacher answer(s) for: " + ", ".join(missing_expected)
        )
    results = []

    for index, cfg in enumerate(CONFIGS, 1):
        candidate = WORK / f"candidate_{index}.pt"
        state = WORK / f"state_{index}.json"
        write_empty_state(state)

        print()
        print("-" * 96)
        print(
            f"Candidate {index}/{len(CONFIGS)}: "
            f"manual={cfg['manual_weight']} "
            f"stability={cfg['stability_weight']} "
            f"lr={cfg['lr']}"
        )

        cmd = [
            sys.executable,
            "online_train.py",
            "--chat-data", str(CHAT_LOG),
            "--base-model", str(BASE_MODEL),
            "--tokenizer", str(TOKENIZER),
            "--output", str(candidate),
            "--state", str(state),
            "--stability-data", str(STABILITY),
            "--stability-weight", str(cfg["stability_weight"]),
            "--learning-rate", str(cfg["lr"]),
            "--manual-weight", str(cfg["manual_weight"]),
            "--trusted-replay-weight", "1",
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            results.append((index, cfg, None, None))
            print(f"[TRAIN FAIL] exit={proc.returncode}")
            continue

        score = evaluate(candidate, expected_learned)
        results.append((index, cfg, candidate, score))
        print(
            f"[EVAL] baseline={score['baseline']}/{len(BASELINE)} "
            f"learned={score['learned']}/{len(LEARNED)} "
            f"unknown={score['unknown']}/{len(UNKNOWN)} "
            f"total={score['total']}/{len(BASELINE)+len(LEARNED)+len(UNKNOWN)} "
            f"fidelity={score['learned_fidelity']:.3f} "
            f"strict={'PASS' if score['strict_pass'] else 'FAIL'}"
        )
        for line in score["details"]:
            print("  " + line)

    valid = [row for row in results if row[3] is not None]
    if not valid:
        raise RuntimeError("No candidate completed training/evaluation.")

    # Prefer strict pass, then maximize teacher-answer fidelity. If several
    # candidates are equally faithful, prefer stronger stability replay and
    # lower manual weight to reduce unnecessary plasticity.
    strict = [row for row in valid if row[3]["strict_pass"]]
    pool = strict if strict else valid
    best = max(
        pool,
        key=lambda row: (
            row[3]["total"],
            row[3]["learned_fidelity"],
            row[3]["baseline"],
            row[3]["unknown"],
            row[1]["stability_weight"],
            -row[1]["manual_weight"],
            -float(row[1]["lr"]),
            -row[0],
        ),
    )

    index, cfg, candidate, score = best
    assert candidate is not None and score is not None

    shutil.copy2(candidate, ONLINE_MODEL)
    shutil.copy2(WORK / f"state_{index}.json", STATE)

    print()
    print("=" * 96)
    print("Selected candidate")
    print("=" * 96)
    print("Index            :", index)
    print("Manual weight    :", cfg["manual_weight"])
    print("Stability weight :", cfg["stability_weight"])
    print("Learning rate    :", cfg["lr"])
    print(
        "Score            :",
        f"{score['total']}/{len(BASELINE)+len(LEARNED)+len(UNKNOWN)}",
    )
    print("Strict pass      :", score["strict_pass"])
    print("Learned fidelity :", f"{score['learned_fidelity']:.3f}")
    print("Installed model  :", ONLINE_MODEL)
    print("Installed state  :", STATE)
    print("Backup           :", backup_dir)
    print()
    print("Next:")
    print("  python eval_multiconcept_incremental_v104.py")

    raise SystemExit(0 if score["strict_pass"] else 2)


if __name__ == "__main__":
    main()
