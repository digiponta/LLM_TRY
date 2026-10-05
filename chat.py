# chat.py
#
# Interactive chat interface for the current LLM_TRY conversational checkpoint.
# Defaults to the v9.4 Nagato SFT model used by the v10.x stable baseline.
#
# v1.5.12 additions:
#   - conservative chat-level Unknown rejection
#   - multiple probe generations
#   - token-confidence / response-agreement checks
#   - malformed / repetitive output detection
#   - fallback response: "未学習です"
#
# Note:
# The v1.4.32/v1.5.1 semantic risk predictors are contrast-routing specific.
# They cannot be applied directly to arbitrary chat prompts.  This file uses
# a chat-level conservative rejection gate inspired by the same selective
# prediction principle.

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import json
import hashlib
import re
import subprocess
import sys
import time
import unicodedata
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LanguageModel
from tokenizer_bpe import Tokenizer
from semantic_proposition_v1090 import (
    add_statement as add_semantic_proposition_statement,
    compose_subject as compose_semantic_proposition_subject,
    load_propositions as load_semantic_propositions,
)
from unified_semantic_bridge_v1090 import (
    sync_all_propositions,
    sync_subject_from_propositions,
)
from subject_keyed_proposition_v1090 import (
    compose_subject_from_index,
    load_subject_index,
    subject_index_dict,
    subject_mapping,
    sync_subject_index,
)
from typed_subject_proposition_v10100 import (
    load_typed_index,
    sync_typed_index,
    typed_index_dict,
    typed_query_lookup,
    typed_subject_mapping,
    typed_subject_rows,
)
from internalized_knowledge_v10100 import (
    internalized_record_for_focus,
    load_internalized_concepts,
    load_internalized_records,
)
from knowledge_state_resolver_v10100 import (
    resolve_knowledge_state,
)
from knowledge_state_dispatcher_v10101 import (
    dispatch_knowledge_state,
)
from truth_state_v10103 import (
    TRUTH_STATES,
    effective_truth_record,
    load_truth_records,
    upsert_truth_record,
)
from truth_aware_dispatch_v10103 import (
    apply_truth_policy,
)
from semantic_knowledge_architecture_v10110 import (
    SemanticKnowledgeArchitecture,
    SemanticKnowledgeConfig,
)
from semantic_knowledge_snapshot_v10111 import (
    snapshot_semantic_knowledge,
)
from knowledge_promotion_v10114 import (
    pending_knowledge_requests,
    promote_knowledge,
)
from knowledge_queue_lifecycle_v10115 import (
    consolidate_legacy_resolved,
    lifecycle_summary,
    mark_concept_verified,
    revoke_concept_verification,
)


DEFAULT_TOKENIZER = "model/tokenizer-v0.7-bpe.json"
DEFAULT_MODEL = "model/model-llm-try-nagato-chat-v94.pt"
DEFAULT_PREVIOUS_ONLINE_MODEL = "model/model-gpu-v1.6.1-online.pt"
DEFAULT_CONCEPT_CALIBRATION = "model/concept-calibration-v1512.pt"
DEFAULT_LEARNING_LOG = "data/chat_history.jsonl"
DEFAULT_LEARNING_STATE = "data/chat_learning_state.json"
DEFAULT_TEACHING_QUEUE = "data/teaching_queue.jsonl"
DEFAULT_KNOWLEDGE_QUEUE = "data/knowledge_queue.jsonl"
DEFAULT_GATE_REVIEW_QUEUE = "data/gate_review_queue.jsonl"
DEFAULT_ONLINE_MODEL = "model/model-gpu-v1.6.2-online.pt"
DEFAULT_ONLINE_TRAINER = "online_train.py"
DEFAULT_RAW_KNOWLEDGE_CORPUS = "data/data-nagato.txt"
DEFAULT_SEMANTIC_KNOWLEDGE = "data/unified_semantic_memory_v1090.jsonl"
DEFAULT_PROPOSITION_STORE = "data/semantic_propositions_v1090.jsonl"
DEFAULT_SUBJECT_INDEX = "data/subject_keyed_propositions_v1090.jsonl"
DEFAULT_TYPED_SUBJECT_INDEX = "data/typed_subject_propositions_v10100.jsonl"
DEFAULT_TRUTH_STORE = "data/truth_state_v10103.jsonl"

USER_PREFIX = "人: "
AI_PREFIX = "AI: "
UNKNOWN_REPLY = "未学習です"

PROJECT_ROOT = Path(__file__).resolve().parent
SIBLING_LLM_GPU_ROOT = PROJECT_ROOT.parent / "LLM_GPU"


def resolve_runtime_path(value: str | Path) -> Path:
    """Resolve runtime assets robustly from LLM_TRY or sibling LLM_GPU.

    Resolution order for relative paths:
      1. current working directory (backward compatible)
      2. LLM_TRY repository root
      3. sibling ../LLM_GPU repository (for shared model assets)
    """
    raw = Path(value)
    if raw.is_absolute():
        return raw

    candidates = [raw, PROJECT_ROOT / raw]
    if raw.parts and raw.parts[0] == "model":
        candidates.append(SIBLING_LLM_GPU_ROOT / raw)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_ROOT / raw).resolve()


@dataclass
class GenerationResult:
    text: str
    token_count: int
    mean_confidence: float
    min_confidence: float
    mean_top2_margin: float


@dataclass(frozen=True)
class CompositeFidelityResult:
    semantic_similarity: float
    lexical_coverage: float
    required_coverage: float
    contradiction: bool
    required_terms: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()

    @property
    def missing_terms(self) -> tuple[str, ...]:
        matched = set(self.matched_terms)
        return tuple(
            term for term in self.required_terms
            if term not in matched
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the current LLM_GPU conversational model."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--concept-calibration",
        default=DEFAULT_CONCEPT_CALIBRATION,
        help=(
            "Optional v1.5.12 concept calibration checkpoint. "
            "Uses train-centroid calibrated space when available."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--learning-log", default=DEFAULT_LEARNING_LOG)
    parser.add_argument("--learning-state", default=DEFAULT_LEARNING_STATE)
    parser.add_argument("--teaching-queue", default=DEFAULT_TEACHING_QUEUE)
    parser.add_argument("--knowledge-queue", default=DEFAULT_KNOWLEDGE_QUEUE)
    parser.add_argument("--gate-review-queue", default=DEFAULT_GATE_REVIEW_QUEUE)
    parser.add_argument(
        "--propositions",
        default=DEFAULT_PROPOSITION_STORE,
        help="Persistent atomic semantic proposition store.",
    )
    parser.add_argument(
        "--subject-index",
        default=DEFAULT_SUBJECT_INDEX,
        help="Derived subject-keyed semantic proposition index.",
    )
    parser.add_argument(
        "--typed-subject-index",
        default=DEFAULT_TYPED_SUBJECT_INDEX,
        help="Derived Subject => Predicate Type => Statement index.",
    )
    parser.add_argument(
        "--truth-store",
        default=DEFAULT_TRUTH_STORE,
        help="Persistent concept truth-state overlay.",
    )
    parser.add_argument(
        "--learn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start with accepted-turn learning capture enabled.",
    )
    parser.add_argument("--online-trainer", default=DEFAULT_ONLINE_TRAINER)
    parser.add_argument("--online-output", default=DEFAULT_ONLINE_MODEL)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=3,
        help="Number of previous turns included in the prompt.",
    )

    # Conservative selective-prediction gate.
    parser.add_argument(
        "--unknown-rejection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable conservative chat-level Unknown rejection.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=3,
        help="Number of generation probes used for agreement.",
    )
    parser.add_argument(
        "--probe-temperature",
        type=float,
        default=0.30,
        help="Sampling temperature for reliability probes.",
    )
    parser.add_argument(
        "--probe-top-k",
        type=int,
        default=10,
        help="Top-k used for reliability probes.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.18,
        help="Minimum mean selected-token probability.",
    )
    parser.add_argument(
        "--min-token-confidence",
        type=float,
        default=0.02,
        help="Minimum probability allowed for any generated token.",
    )
    parser.add_argument(
        "--min-mean-margin",
        type=float,
        default=0.01,
        help="Minimum mean top1-vs-top2 probability margin.",
    )
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.35,
        help="Minimum mean textual agreement between probe responses.",
    )
    parser.add_argument(
        "--min-semantic-agreement",
        type=float,
        default=0.82,
        help="Minimum mean cosine similarity between probe response vectors.",
    )
    parser.add_argument(
        "--min-internalized-fidelity",
        type=float,
        default=0.90,
        help="Minimum teacher semantic similarity for INTERNALIZED output.",
    )
    parser.add_argument(
        "--min-internalized-lexical-coverage",
        type=float,
        default=0.45,
        help="Minimum teacher character-bigram coverage.",
    )
    parser.add_argument(
        "--min-internalized-required-coverage",
        type=float,
        default=0.50,
        help="Minimum required-content term coverage.",
    )
    parser.add_argument(
        "--semantic-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable question-answer semantic consistency checks.",
    )
    parser.add_argument(
        "--history-contamination-margin",
        type=float,
        default=0.05,
        help=(
            "Reject if the answer is closer to a previous user turn than "
            "the current turn by this cosine-similarity margin."
        ),
    )
    parser.add_argument(
        "--show-risk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show confidence/agreement diagnostics.",
    )
    return parser.parse_args()




def normalize_pair_text(text: str) -> str:
    return " ".join(text.strip().split())


def pair_fingerprint(user: str, answer: str) -> str:
    payload = normalize_pair_text(user) + "\n" + normalize_pair_text(answer)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trusted_fingerprints_from_log(path: Path) -> set[str]:
    fingerprints: set[str] = set()
    if not path.exists():
        return fingerprints

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        source = str(row.get("source", ""))
        if source not in ("chat-manual", "chat-approved"):
            continue
        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if user and answer:
            fingerprints.add(pair_fingerprint(user, answer))
    return fingerprints


def initialize_learning_state_if_missing(
    state_path: Path,
    learning_log: Path,
    assume_existing_trained: bool,
) -> int:
    if state_path.exists():
        return 0

    fingerprints = (
        trusted_fingerprints_from_log(learning_log)
        if assume_existing_trained
        else set()
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v1.6.2",
        "trained_fingerprints": sorted(fingerprints),
    }
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(fingerprints)


def mark_pair_for_retraining(
    state_path: Path,
    user_text: str,
    assistant_text: str,
) -> bool:
    if not state_path.exists():
        return False

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    fingerprints = {
        str(x) for x in data.get("trained_fingerprints", [])
    }
    fp = pair_fingerprint(user_text, assistant_text)
    if fp not in fingerprints:
        return False

    fingerprints.remove(fp)
    data["trained_fingerprints"] = sorted(fingerprints)
    data["version"] = data.get("version", "v1.6.2")
    state_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def choose_startup_model(
    requested_model: str,
    online_output: str = DEFAULT_ONLINE_MODEL,
    learning_state: str = DEFAULT_LEARNING_STATE,
) -> Path:
    """Choose the adaptive checkpoint only when training state proves it exists.

    Explicit --model always wins.  For the default v10.x startup, a local
    online checkpoint is resumed only when chat_learning_state.json records at
    least one consumed trusted fingerprint.  This avoids accidentally loading
    an unrelated legacy online checkpoint.
    """
    requested = resolve_runtime_path(requested_model)
    if requested_model != DEFAULT_MODEL:
        return requested

    state_path = resolve_runtime_path(learning_state)
    online_path = resolve_runtime_path(online_output)
    if not state_path.exists() or not online_path.exists():
        return requested

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return requested

    fingerprints = state.get("trained_fingerprints", [])
    if isinstance(fingerprints, list) and fingerprints:
        return online_path

    return requested

def append_learning_pair(
    path: Path,
    user_text: str,
    assistant_text: str,
    source: str = "chat-auto",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "user": user_text,
        "assistant": assistant_text,
        "source": source,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_route_record(
    path: Path,
    resolution: str,
    action: str,
    user_text: str,
    candidate_answer: str,
    reason: str,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = pair_fingerprint(
        resolution + "\n" + reason,
        user_text,
    )

    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                old = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if old.get("fingerprint") == fingerprint:
                return False

    row = {
        "fingerprint": fingerprint,
        "resolution": resolution,
        "action": action,
        "user": user_text,
        "candidate_answer": candidate_answer,
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def route_resolution_action(
    args: argparse.Namespace,
    resolution: str,
    action: str,
    user_text: str,
    candidate_answer: str,
    reason: str,
) -> str:
    if resolution == "LEARNING_GAP":
        path = Path(args.teaching_queue)
        added = append_route_record(
            path, resolution, action, user_text, candidate_answer, reason
        )
        return (
            f"queued teaching candidate -> {path}"
            if added else
            f"teaching candidate already queued -> {path}"
        )

    if resolution == "UNKNOWN_KNOWLEDGE":
        path = Path(args.knowledge_queue)
        added = append_route_record(
            path, resolution, action, user_text, candidate_answer, reason
        )
        return (
            f"queued knowledge request -> {path}"
            if added else
            f"knowledge request already queued -> {path}"
        )

    if resolution == "GATE_REVIEW":
        path = Path(args.gate_review_queue)
        added = append_route_record(
            path, resolution, action, user_text, candidate_answer, reason
        )
        return (
            f"queued gate review -> {path}"
            if added else
            f"gate review already queued -> {path}"
        )

    if resolution == "INPUT_REJECT":
        return "request rephrase"

    return "normal response"


def validate_teaching_answer(
    question: str,
    corrected: str,
) -> tuple[bool, str]:
    intent, slots = classify_intent_and_slots(question)

    if intent == "definition" and slots:
        slot = slots[0]
        if definition_is_echo_only(slot, corrected):
            return (
                False,
                "definition teaching answer is only an echo; "
                "teach category/function/property, not only a synonym",
            )
        if definition_is_uninformative(slot, corrected):
            return (
                False,
                "definition teaching answer is uninformative; "
                "include category, function, or distinguishing property",
            )

    concept_ok, concept_reason = answer_concept_consistency(
        question,
        corrected,
    )
    if not concept_ok:
        return False, concept_reason

    return True, "teaching answer accepted"


def teaching_hint(question: str) -> str:
    focus = extract_definition_focus(question)
    if not focus:
        return "Provide a concrete, correct answer with the key concept and function."

    key = focus.lower()
    examples = {
        "ai": "AIは人間の知的な処理をコンピュータで実現する技術です。",
        "人工知能": "人工知能は人間の知的な処理をコンピュータで実現する技術です。",
        "cpu": "CPUは汎用的な命令実行と制御処理を担当する中央処理装置です。",
        "gpu": "GPUは多数の演算を並列に処理するプロセッサです。",
        "llm": "LLMは大量のテキストから学習し、言語を扱う大規模言語モデルです。",
        "大規模言語モデル": "大規模言語モデルは大量のテキストから学習し、言語を扱うモデルです。",
    }
    return examples.get(
        key,
        f"{focus}について、分類・機能・特徴のいずれかを含む説明を教えてください。",
    )


def learning_log_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def trusted_pairs_from_log(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not path.exists():
        return pairs

    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if str(row.get("source", "")) not in (
            "chat-manual", "chat-approved", "chat-recovery"
        ):
            continue
        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if not user or not answer:
            continue
        fp = pair_fingerprint(user, answer)
        if fp in seen:
            continue
        seen.add(fp)
        pairs.append((user, answer))
    return pairs


def trained_known_concepts(
    learning_log: Path,
    learning_state: Path,
) -> set[str]:
    """Return definition focuses from trusted pairs already consumed by /train.

    A concept is promoted only when the exact trusted pair fingerprint appears
    in the training state. Merely teaching a pair is not enough.
    """
    if not learning_log.exists() or not learning_state.exists():
        return set()

    try:
        state = json.loads(learning_state.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()

    trained = {str(x) for x in state.get("trained_fingerprints", [])}
    if not trained:
        return set()

    concepts: set[str] = set()
    for raw in learning_log.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue

        source = str(row.get("source", ""))
        if source not in ("chat-manual", "chat-approved", "chat-recovery"):
            continue

        user = str(row.get("user", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if not user or not answer:
            continue

        if pair_fingerprint(user, answer) not in trained:
            continue

        focus = extract_concept_query_focus(user)
        if not focus:
            focus = extract_definition_focus(user) or ""
        if focus:
            concepts.add(focus.lower())

    return concepts


def recover_forgotten_pairs_from_queue(
    teaching_queue: Path,
    learning_log: Path,
    learning_state: Path,
    target_question: str | None = None,
) -> tuple[int, int, int, list[tuple[str, str, str]]]:
    if not teaching_queue.exists():
        return 0, 0, 0, []

    queued_questions: list[str] = []
    seen_questions: set[str] = set()

    for raw in teaching_queue.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if str(row.get("resolution", "")) != "LEARNING_GAP":
            continue
        if str(row.get("status", "pending")) == "resolved":
            continue

        question = str(row.get("user", "")).strip()
        if target_question is not None:
            if (
                normalize_pair_text(question).lower()
                != normalize_pair_text(target_question).lower()
            ):
                continue
        qn = normalize_pair_text(question).lower()
        if question and qn not in seen_questions:
            seen_questions.add(qn)
            queued_questions.append(question)

    if not queued_questions:
        return 0, 0, 0, []

    trusted_pairs = trusted_pairs_from_log(learning_log)
    reactivated = 0
    matched = 0
    rejected_old_teachers = 0
    selections: list[tuple[str, str, str]] = []

    for question in queued_questions:
        qn = normalize_pair_text(question).lower()
        queued_intent, queued_slots = classify_intent_and_slots(question)
        queued_focus = (
            _clean_slot(queued_slots[0]).lower()
            if queued_intent == "definition" and queued_slots
            else None
        )

        best: tuple[str, str] | None = None
        best_rank = -1.0

        for user, answer in trusted_pairs:
            trusted_intent, trusted_slots = classify_intent_and_slots(user)
            trusted_focus = (
                _clean_slot(trusted_slots[0]).lower()
                if trusted_intent == "definition" and trusted_slots
                else None
            )

            if queued_intent == "definition":
                if trusted_intent != "definition":
                    continue
                if queued_focus != trusted_focus:
                    continue

            un = normalize_pair_text(user).lower()
            similarity = response_similarity(qn, un)
            if qn == un:
                similarity = 1.0
            if similarity < 0.80:
                continue

            historical_ok, _ = validate_teaching_answer(user, answer)
            queued_ok, _ = validate_teaching_answer(question, answer)
            if not historical_ok or not queued_ok:
                rejected_old_teachers += 1
                continue

            informativeness = min(
                len(normalize_pair_text(answer)) / 200.0,
                0.20,
            )
            rank = similarity + informativeness
            if rank > best_rank:
                best_rank = rank
                best = (user, answer)

        if best is None:
            selections.append((question, "", "no valid trusted teacher"))
            continue

        matched += 1
        selections.append((question, best[1], f"trusted question={best[0]}"))
        if mark_pair_for_retraining(
            learning_state,
            best[0],
            best[1],
        ):
            append_learning_pair(
                learning_log,
                best[0],
                best[1],
                source="chat-recovery",
            )
            reactivated += 1

    return matched, reactivated, rejected_old_teachers, selections


def resolve_route_queue(
    queue_path: Path,
    question: str,
    resolution: str,
) -> int:
    """Mark matching routed work items resolved after trusted teaching."""
    if not queue_path.exists():
        return 0

    target = normalize_pair_text(question).lower()
    changed = 0
    rows: list[dict] = []

    for raw in queue_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue

        queued_question = normalize_pair_text(str(row.get("user", ""))).lower()
        if (
            str(row.get("resolution", "")) == resolution
            and queued_question == target
            and str(row.get("status", "pending")) != "resolved"
        ):
            row["status"] = "resolved"
            row["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            changed += 1

        rows.append(row)

    if changed:
        queue_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
    return changed


def resolve_teaching_queue(
    teaching_queue: Path,
    question: str,
) -> int:
    return resolve_route_queue(
        teaching_queue,
        question,
        "LEARNING_GAP",
    )


def run_online_training(
    args: argparse.Namespace,
    model_path: Path,
) -> Path | None:
    trainer = Path(args.online_trainer)
    learning_log = Path(args.learning_log)
    output = Path(args.online_output)

    if not trainer.exists():
        print(f"[online trainer not found: {trainer}]")
        return None
    if learning_log_count(learning_log) < 2:
        print("[need at least 2 learning pairs before /train]")
        return None

    cmd = [
        sys.executable,
        str(trainer),
        "--chat-data", str(learning_log),
        "--base-model", str(model_path),
        "--tokenizer", str(resolve_runtime_path(args.tokenizer)),
        "--output", str(output),
        "--state", str(args.learning_state),
    ]
    print("[starting incremental training]")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode == 3:
        print("[no new trusted pairs; training skipped]")
        return None
    if completed.returncode != 0:
        print(f"[training failed: exit={completed.returncode}]")
        return None
    if not output.exists():
        print(f"[training finished but checkpoint not found: {output}]")
        return None
    return output

def _slot_overlap(a: list[str], b: list[str]) -> bool:
    aa = {x.lower() for x in a}
    bb = {x.lower() for x in b}
    return bool(aa & bb)


def select_relevant_history(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> List[Tuple[str, str]]:
    """v1.5.5 minimal-context policy.

    definition  : no raw history
    comparison  : no raw history
    greeting    : at most one previous greeting
    general     : at most one highly similar previous user turn
    """
    if history_turns <= 0 or not history:
        return []

    current_intent, _ = classify_intent_and_slots(user_text)

    if current_intent in ("definition", "comparison"):
        return []

    if current_intent == "greeting":
        for old_user, old_ai in reversed(history):
            old_intent, _ = classify_intent_and_slots(old_user)
            if old_intent == "greeting":
                return [(old_user, old_ai)]
        return []

    # General question: keep only one closely related prior user turn.
    # Character bigram similarity is deliberately conservative and cheap.
    best: Tuple[str, str] | None = None
    best_score = 0.0
    for old_user, old_ai in history:
        score = response_similarity(user_text, old_user)
        if score > best_score:
            best_score = score
            best = (old_user, old_ai)

    if best is not None and best_score >= 0.55:
        return [best]

    return []


IDENTITY_QUERY_ALIASES = {
    "貴方は": "あなたは誰ですか",
    "あなたは": "あなたは誰ですか",
    "長門": "あなたは誰ですか",
    "長門有希": "あなたは誰ですか",
    "長門とは": "あなたは誰ですか",
    "長門有希とは": "あなたは誰ですか",
}


def normalize_identity_query(question: str) -> str:
    """Map short identity variants to the stable canonical identity prompt."""
    return IDENTITY_QUERY_ALIASES.get(question.strip(), question)


def build_prompt(
    history: List[Tuple[str, str]],
    user_text: str,
    history_turns: int,
) -> tuple[str, List[Tuple[str, str]]]:
    chunks: List[str] = []
    selected = select_relevant_history(
        history=history,
        user_text=user_text,
        history_turns=history_turns,
    )

    for old_user, old_ai in selected:
        chunks.append(
            f"{USER_PREFIX}{old_user}\n"
            f"{AI_PREFIX}{old_ai}\n"
        )

    chunks.append(f"{USER_PREFIX}{user_text}\n{AI_PREFIX}")
    return "".join(chunks), selected


def _clean_reply(reply: str) -> str:
    for marker in ("\n人:", "\nAI:", "\n"):
        if marker in reply:
            reply = reply.split(marker, 1)[0]
    return reply.strip()


@torch.no_grad()
def generate_reply(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    seed: int | None = None,
) -> GenerationResult:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = list(prompt_ids)
    response_ids: List[int] = []

    confidences: List[float] = []
    margins: List[float] = []

    model.eval()
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        context = generated[-model.context_length:]
        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        raw_logits = model(x)[0, -1, :].clone()

        if repetition_penalty != 1.0:
            for token_id in set(response_ids):
                if 0 <= token_id < raw_logits.numel():
                    if raw_logits[token_id] >= 0:
                        raw_logits[token_id] /= repetition_penalty
                    else:
                        raw_logits[token_id] *= repetition_penalty

        # Confidence is always measured on the untempered model distribution.
        raw_probs = F.softmax(raw_logits, dim=-1)
        top2_values, _ = torch.topk(raw_probs, k=2)

        if temperature <= 0:
            next_id = int(torch.argmax(raw_logits).item())
        else:
            logits = raw_logits / temperature
            if 0 < top_k < logits.numel():
                values, indices = torch.topk(logits, top_k)
                probs = F.softmax(values, dim=-1)
                selected = torch.multinomial(probs, 1)
                next_id = int(indices[selected].item())
            else:
                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

        if next_id == tokenizer.eos_id:
            break

        confidences.append(float(raw_probs[next_id].item()))
        margins.append(float((top2_values[0] - top2_values[1]).item()))

        generated.append(next_id)
        response_ids.append(next_id)

        decoded = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        if "\n" in decoded:
            break

    reply = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )
    reply = _clean_reply(reply)

    mean_conf = (
        sum(confidences) / len(confidences)
        if confidences else 0.0
    )
    min_conf = min(confidences) if confidences else 0.0
    mean_margin = (
        sum(margins) / len(margins)
        if margins else 0.0
    )

    return GenerationResult(
        text=reply,
        token_count=len(response_ids),
        mean_confidence=mean_conf,
        min_confidence=min_conf,
        mean_top2_margin=mean_margin,
    )


def _normalize_for_similarity(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[。、,.!?！？:：;；「」『』（）()\[\]{}]", "", text)
    return text


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    text = _normalize_for_similarity(text)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text)-n+1)}


def response_similarity(a: str, b: str) -> float:
    aa = _char_ngrams(a)
    bb = _char_ngrams(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def mean_pairwise_agreement(results: List[GenerationResult]) -> float:
    if len(results) < 2:
        return 1.0
    values: List[float] = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            values.append(
                response_similarity(results[i].text, results[j].text)
            )
    return sum(values) / len(values) if values else 1.0


def malformed_or_unstable(text: str) -> bool:
    if not text:
        return True

    # Unicode replacement character is a strong malformed-output signal.
    if "�" in text:
        return True

    # Reject pathological character/token repetition.
    compact = _normalize_for_similarity(text)
    if len(compact) >= 8:
        for n in (1, 2, 3, 4):
            chunks = [
                compact[i:i+n]
                for i in range(0, len(compact)-n+1)
            ]
            if chunks:
                most = max(chunks.count(x) for x in set(chunks))
                if most / len(chunks) > 0.45:
                    return True

    return False


@torch.no_grad()
def semantic_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    block_index: int = 4,
) -> torch.Tensor:
    """Mean pooled hidden representation at block5 (index 4)."""
    ids = tokenizer.encode(text, add_bos=True)
    if not ids:
        return torch.zeros(model.d_model, device=next(model.parameters()).device)

    ids = ids[-model.context_length:]
    device = next(model.parameters()).device
    x_ids = torch.tensor([ids], dtype=torch.long, device=device)

    x = model.embedding(x_ids)
    if model.position_embedding is not None:
        positions = torch.arange(x_ids.shape[1], device=device)
        x = x + model.position_embedding(positions).unsqueeze(0)

    last = min(block_index, len(model.blocks)-1)
    for i in range(last + 1):
        x = model.blocks[i](x)

    # Exclude BOS when possible.
    token_states = x[0, 1:, :] if x.shape[1] > 1 else x[0]
    v = token_states.mean(dim=0)
    return F.normalize(v, dim=0)


@torch.no_grad()
def mean_pairwise_semantic_agreement(
    model: LanguageModel,
    tokenizer: Tokenizer,
    results: List[GenerationResult],
) -> float:
    if len(results) < 2:
        return 1.0

    vectors = [
        semantic_vector(model, tokenizer, result.text)
        for result in results
    ]
    values: List[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            values.append(float(torch.dot(vectors[i], vectors[j]).item()))
    return sum(values) / len(values) if values else 1.0


@torch.no_grad()
def internalized_teacher_fidelity(
    model: LanguageModel,
    tokenizer: Tokenizer,
    generated_answer: str,
    teacher_answer: str,
) -> float:
    if not generated_answer.strip() or not teacher_answer.strip():
        return 0.0
    generated_vector = semantic_vector(
        model,
        tokenizer,
        generated_answer,
    )
    teacher_vector = semantic_vector(
        model,
        tokenizer,
        teacher_answer,
    )
    return float(torch.dot(generated_vector, teacher_vector).item())


def _teacher_lexical_coverage(
    generated_answer: str,
    teacher_answer: str,
) -> float:
    teacher = _char_ngrams(teacher_answer, n=2)
    generated = _char_ngrams(generated_answer, n=2)
    if not teacher:
        return 0.0
    return len(teacher & generated) / len(teacher)


def _required_teacher_terms(
    teacher_answer: str,
    concept: str = "",
) -> tuple[str, ...]:
    text = _normalize_for_similarity(teacher_answer)
    concept_norm = _normalize_for_similarity(concept)
    if concept_norm and text.startswith(concept_norm):
        text = text[len(concept_norm):]
    text = re.sub(r"^(?:は|とは)", "", text)
    text = re.sub(r"(?:である|です|だ)$", "", text)

    parts = re.split(
        r"(?:を|に|で|が|は|と|して|する|行う|利用|ため|の)",
        text,
    )
    stop = {
        "もの", "こと", "ため", "これ", "それ",
        "ある", "いる", "なる", "できる",
    }
    terms: list[str] = []
    for part in parts:
        term = part.strip()
        if len(term) < 2 or term in stop:
            continue
        if term not in terms:
            terms.append(term)

    # Preserve the terminal category noun when it is informative.
    category_match = re.search(
        r"([一-龯々ァ-ヶーA-Za-z0-9]{2,12})(?:である|です|だ)$",
        teacher_answer.strip(),
    )
    if category_match:
        category = category_match.group(1)
        if (
            category != concept
            and category not in terms
            and len(category) >= 2
        ):
            terms.append(category)

    return tuple(terms[:8])


def _required_content_coverage(
    generated_answer: str,
    teacher_answer: str,
    concept: str = "",
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    required = _required_teacher_terms(
        teacher_answer,
        concept=concept,
    )
    if not required:
        return 1.0, (), ()
    generated_norm = _normalize_for_similarity(generated_answer)
    matched = tuple(
        term for term in required
        if _normalize_for_similarity(term) in generated_norm
    )
    return len(matched) / len(required), required, matched


def _polarity_contradiction(
    generated_answer: str,
    teacher_answer: str,
) -> bool:
    negative_patterns = (
        r"ではない", r"でない", r"しない", r"できない",
        r"行わない", r"利用しない", r"不可能", r"存在しない",
    )
    teacher_negative = any(
        re.search(pattern, teacher_answer)
        for pattern in negative_patterns
    )
    generated_negative = any(
        re.search(pattern, generated_answer)
        for pattern in negative_patterns
    )
    return teacher_negative != generated_negative


@torch.no_grad()
def composite_internalized_teacher_fidelity(
    model: LanguageModel,
    tokenizer: Tokenizer,
    generated_answer: str,
    teacher_answer: str,
    concept: str = "",
) -> CompositeFidelityResult:
    semantic = internalized_teacher_fidelity(
        model=model,
        tokenizer=tokenizer,
        generated_answer=generated_answer,
        teacher_answer=teacher_answer,
    )
    lexical = _teacher_lexical_coverage(
        generated_answer,
        teacher_answer,
    )
    required_coverage, required, matched = (
        _required_content_coverage(
            generated_answer,
            teacher_answer,
            concept=concept,
        )
    )
    contradiction = _polarity_contradiction(
        generated_answer,
        teacher_answer,
    )
    return CompositeFidelityResult(
        semantic_similarity=semantic,
        lexical_coverage=lexical,
        required_coverage=required_coverage,
        contradiction=contradiction,
        required_terms=required,
        matched_terms=matched,
    )


def apply_composite_internalized_fidelity_policy(
    accepted: bool,
    result: CompositeFidelityResult,
    *,
    semantic_threshold: float,
    lexical_threshold: float,
    required_threshold: float,
) -> tuple[bool, str]:
    failures: list[str] = []
    if result.semantic_similarity < semantic_threshold:
        failures.append(
            "semantic "
            f"{result.semantic_similarity:.3f} < {semantic_threshold:.3f}"
        )
    if result.lexical_coverage < lexical_threshold:
        failures.append(
            "lexical "
            f"{result.lexical_coverage:.3f} < {lexical_threshold:.3f}"
        )
    if result.required_coverage < required_threshold:
        failures.append(
            "required "
            f"{result.required_coverage:.3f} < {required_threshold:.3f}"
        )
    if result.contradiction:
        failures.append("contradiction detected")

    if failures:
        return False, "composite teacher fidelity: " + "; ".join(failures)
    if not accepted:
        return False, ""
    return True, "composite teacher fidelity passed"


SLOT_ALIASES = {
    "ai": ("ai", "人工知能", "artificial intelligence"),
    "人工知能": ("人工知能", "ai", "artificial intelligence"),
    "llm": ("llm", "大規模言語モデル", "large language model", "言語モデル"),
    "大規模言語モデル": ("大規模言語モデル", "llm", "large language model", "言語モデル"),
    "cpu": ("cpu", "中央処理装置", "central processing unit"),
    "gpu": ("gpu", "画像処理装置", "graphics processing unit"),
    "cuda": ("cuda",),
}


STRICT_SYNONYMS = {
    "ai": ("ai", "人工知能", "artificial intelligence"),
    "人工知能": ("人工知能", "ai", "artificial intelligence"),
    "llm": ("llm", "大規模言語モデル", "large language model"),
    "大規模言語モデル": ("大規模言語モデル", "llm", "large language model"),
    "cpu": ("cpu", "中央処理装置", "central processing unit"),
    "gpu": ("gpu", "graphics processing unit"),
    "cuda": ("cuda",),
}

DEFINITION_CATEGORIES = {
    "量子力学": ("理論", "微視的", "状態", "重ね合わせ", "測定"),
    "量子コンピュータ": ("量子", "計算", "量子ビット", "重ね合わせ"),
    "コンピュータ": ("計算機", "装置", "計算", "情報処理"),
    "semantic": ("意味", "意味論", "セマンティック"),
    "セマンティック": ("意味", "意味論", "セマンティック"),
    "llm": ("言語モデル",),
    "大規模言語モデル": ("言語モデル",),
    "gpu": ("画像処理装置", "プロセッサ", "処理装置"),
    "cpu": ("中央処理装置", "プロセッサ", "処理装置"),
    "ai": ("技術", "システム", "人工知能"),
    "人工知能": ("技術", "システム"),
}


def strict_synonyms(slot: str) -> tuple[str, ...]:
    key = slot.lower()
    return STRICT_SYNONYMS.get(
        key,
        STRICT_SYNONYMS.get(slot, (slot,)),
    )


def definition_categories(slot: str) -> tuple[str, ...]:
    key = slot.lower()
    return DEFINITION_CATEGORIES.get(
        key,
        DEFINITION_CATEGORIES.get(slot, ()),
    )


KNOWN_ENTITY_TOKENS = {
    "AI", "LLM", "CPU", "GPU", "CUDA",
}

CALIBRATION_PROJECTION = None
CALIBRATION_CENTROIDS = None
CALIBRATION_CONCEPTS: list[str] = []
CALIBRATION_INFO: dict = {}


class ChatConceptProjection(nn.Module):
    """Inference-only projection compatible with v1.5.12 checkpoints."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def load_concept_calibration(path: Path, device: torch.device) -> bool:
    global CALIBRATION_PROJECTION
    global CALIBRATION_CENTROIDS
    global CALIBRATION_CONCEPTS
    global CALIBRATION_INFO

    if not path.exists():
        CALIBRATION_PROJECTION = None
        CALIBRATION_CENTROIDS = None
        CALIBRATION_CONCEPTS = []
        CALIBRATION_INFO = {"loaded": False, "path": str(path)}
        return False

    ckpt = torch.load(str(path), map_location=device)

    if ckpt.get("centroid_source") != "train":
        raise ValueError(
            "Concept calibration rejected: centroid_source must be 'train'."
        )
    if ckpt.get("holdout_used_for_centroid", True):
        raise ValueError(
            "Concept calibration rejected: holdout must not be used for centroid fit."
        )

    projection = ChatConceptProjection(
        int(ckpt["input_dim"]),
        int(ckpt["hidden_dim"]),
        int(ckpt["output_dim"]),
    ).to(device)
    projection.load_state_dict(ckpt["projection_state"])
    projection.eval()

    CALIBRATION_PROJECTION = projection
    CALIBRATION_CENTROIDS = F.normalize(
        ckpt["centroids"].to(device), dim=-1
    )
    CALIBRATION_CONCEPTS = list(ckpt["concepts"])
    CALIBRATION_INFO = {
        "loaded": True,
        "path": str(path),
        "version": ckpt.get("version"),
        "evaluation_protocol": ckpt.get("evaluation_protocol"),
        "centroid_source": ckpt.get("centroid_source"),
        "holdout_used_for_centroid": ckpt.get("holdout_used_for_centroid"),
    }
    return True


CONCEPT_ANCHORS = {
    "AI": (
        "AIは人工知能です",
        "人工知能は人間の知的処理を機械で実現する技術です",
    ),
    "LLM": (
        "LLMは大規模言語モデルです",
        "大規模言語モデルは文章を理解し生成するモデルです",
    ),
    "CPU": (
        "CPUは中央処理装置です",
        "CPUは汎用的な命令実行と制御処理を担当します",
    ),
    "GPU": (
        "GPUは画像処理装置です",
        "GPUは大量の並列計算を得意とする演算装置です",
    ),
    "CUDA": (
        "CUDAはNVIDIAのGPU向け汎用計算技術です",
        "CUDAはGPUで計算を実行するための技術です",
    ),
}


@torch.no_grad()
def concept_prototype(
    model: LanguageModel,
    tokenizer: Tokenizer,
    concept: str,
) -> torch.Tensor | None:
    key = _clean_slot(concept).upper()
    anchors = CONCEPT_ANCHORS.get(key)
    if not anchors:
        return None
    vectors = [semantic_vector(model, tokenizer, text) for text in anchors]
    return F.normalize(torch.stack(vectors, dim=0).mean(dim=0), dim=0)


def answer_semantic_spans(answer: str) -> list[str]:
    spans = [
        s.strip()
        for s in re.split(r"[。！？!?、,]|(?:一方)|(?:対して)", answer)
        if s.strip()
    ]
    if answer.strip() and answer.strip() not in spans:
        spans.append(answer.strip())
    return spans


@torch.no_grad()
def concept_slot_match(
    model: LanguageModel,
    tokenizer: Tokenizer,
    slot: str,
    answer: str,
    min_similarity: float = 0.82,
    min_margin: float = 0.005,
) -> tuple[bool, float, float]:
    """Concept-aware fallback.

    v1.5.12 prefers the calibrated train-centroid space. If no compatible
    checkpoint is loaded, it falls back to the raw prototype method.
    """
    target = _clean_slot(slot).upper()

    if (
        CALIBRATION_PROJECTION is not None
        and CALIBRATION_CENTROIDS is not None
        and target in CALIBRATION_CONCEPTS
    ):
        target_idx = CALIBRATION_CONCEPTS.index(target)
        best_score = -1.0
        best_margin = -1.0

        for span in answer_semantic_spans(answer):
            raw = semantic_vector(model, tokenizer, span)
            z = CALIBRATION_PROJECTION(raw.unsqueeze(0))[0]
            sims = z @ CALIBRATION_CENTROIDS.T

            target_score = float(sims[target_idx].item())
            mask = torch.ones(
                len(CALIBRATION_CONCEPTS),
                dtype=torch.bool,
                device=sims.device,
            )
            mask[target_idx] = False
            other_best = float(sims[mask].max().item())
            margin = target_score - other_best

            if margin > best_margin:
                best_score = target_score
                best_margin = margin

        ok = best_margin >= 0.0
        return ok, best_score, best_margin

    target_proto = concept_prototype(model, tokenizer, target)
    if target_proto is None:
        return False, -1.0, -1.0

    other_protos = []
    for concept in CONCEPT_ANCHORS:
        if concept == target:
            continue
        p = concept_prototype(model, tokenizer, concept)
        if p is not None:
            other_protos.append(p)

    best_sim = -1.0
    best_margin = -1.0
    for span in answer_semantic_spans(answer):
        sv = semantic_vector(model, tokenizer, span)
        target_sim = float(torch.dot(target_proto, sv).item())
        other_best = max(
            (float(torch.dot(p, sv).item()) for p in other_protos),
            default=-1.0,
        )
        margin = target_sim - other_best
        if target_sim > best_sim or (
            abs(target_sim - best_sim) < 1e-9 and margin > best_margin
        ):
            best_sim = target_sim
            best_margin = margin

    ok = best_sim >= min_similarity and best_margin >= min_margin
    return ok, best_sim, best_margin


def _clean_slot(slot: str) -> str:
    slot = slot.strip()
    slot = re.sub(r"(?:を|は|が|の)$", "", slot)
    return slot.strip()


def slot_aliases(slot: str) -> tuple[str, ...]:
    key = _clean_slot(slot).lower()
    return SLOT_ALIASES.get(key, (key,))


def slot_present(slot: str, answer: str) -> bool:
    answer_lower = answer.lower()
    return any(alias.lower() in answer_lower for alias in slot_aliases(slot))


def extract_suspicious_entities(text: str) -> list[str]:
    """Extract novel-looking identifiers whose identity should survive generation."""
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_+.#-]{1,39}", text)
    entities: list[str] = []
    for token in candidates:
        upper = token.upper()
        if upper in KNOWN_ENTITY_TOKENS:
            continue

        # Conservative: require digits/hyphen, or mixed-case identifier shape.
        suspicious = (
            any(ch.isdigit() for ch in token)
            or "-" in token
            or (
                any(ch.islower() for ch in token)
                and any(ch.isupper() for ch in token[1:])
                and len(token) >= 5
            )
        )
        if suspicious and token not in entities:
            entities.append(token)
    return entities


def extract_definition_focus(text: str) -> str | None:
    """Extract focus from Japanese definition prompts such as 'GPUとは'."""
    compact = text.strip()
    patterns = [
        r"^\s*([A-Za-z0-9_+.#\-]{2,40})\s*(?:とは|って|とは何|って何)\s*[、,。．！？!?？?]*\s*$",
        r"^\s*([^\s、。！？?]{1,40})\s*(?:とは|って)\s*[、,。．！？!?？?]*\s*$",
    ]
    for p in patterns:
        m = re.search(p, compact, flags=re.IGNORECASE)
        if m:
            return _clean_slot(m.group(1))
    return None



RUNTIME_TRAILING_PUNCTUATION = "、，,。．.!！?？:：;；"


def normalize_runtime_input(text: str) -> str:
    """Normalize harmless surface punctuation before runtime routing.

    This deliberately removes only sentence-final punctuation and applies NFKC.
    Semantic content and internal punctuation are preserved.
    """
    normalized = unicodedata.normalize("NFKC", text).strip()
    return normalized.rstrip(RUNTIME_TRAILING_PUNCTUATION).strip()


def input_quality_check(text: str) -> tuple[bool, str]:
    q = text.strip()
    if not q:
        return False, "empty input"

    compact = _normalize_for_similarity(q)

    # Bare interrogative/function words have no subject or target concept.
    # Do not let the language model guess a topic from prior training bias.
    bare_interrogatives = {
        "とは", "って", "なに", "何", "何ですか", "教えて",
        "説明して", "について", "は", "を", "の",
    }
    if q.rstrip("、。！？?! ").strip() in bare_interrogatives:
        return False, "missing query subject"

    # Repeated interrogative fragments such as "とはとは".
    if re.search(r"(とは){2,}|(って){2,}", q):
        return False, "malformed repeated intent"

    # Obvious typo pattern observed in regression: a known technical acronym
    # followed by an unrelated single kanji instead of an interrogative form.
    if re.fullmatch(r"(AI|CPU|GPU|LLM|CUDA)[一-龯]", q, flags=re.IGNORECASE):
        return False, "malformed technical-term suffix"

    # Multiple bare known entities plus a broken definition marker are
    # ambiguous rather than a well-formed comparison/definition request.
    known_hits = re.findall(r"\b(?:AI|CPU|GPU|LLM|CUDA)\b", q, flags=re.IGNORECASE)
    if len(set(x.upper() for x in known_hits)) >= 2 and "とは" in q:
        if not re.search(r"(違い|比較|差)", q):
            return False, "incoherent multi-concept input"

    # Very short garbage around a known acronym should not be auto-corrected.
    if len(compact) <= 6 and re.search(r"(AI|CPU|GPU|LLM|CUDA)", q, flags=re.IGNORECASE):
        if not (
            extract_definition_focus(q)
            or re.search(r"(について|説明|教えて|違い|比較)", q)
        ):
            # Bare acronym is still allowed as a general query.
            if compact.upper() not in {"AI", "CPU", "GPU", "LLM", "CUDA"}:
                return False, "malformed short technical query"

    return True, "input accepted"


def classify_intent_and_slots(question: str) -> tuple[str, list[str]]:
    q = question.strip()
    q_lower = q.lower()

    greeting_patterns = (
        "こんにちは", "こんばんは", "おはよう", "お疲れ", "はじめまして",
        "やあ", "hello", "hi",
    )
    if any(x in q_lower for x in greeting_patterns):
        return "greeting", []

    # Comparison: "AとBの違い", "AとBを比較", "AとBの差"
    m = re.search(
        r"^\s*([^\s、。！？?と]{1,30})\s*と\s*([^\s、。！？?]{1,30}?)"
        r"\s*(?:を|の)?(?:違い|差|比較|違う点)",
        q,
        flags=re.IGNORECASE,
    )
    if m:
        return "comparison", [
            _clean_slot(m.group(1)),
            _clean_slot(m.group(2)),
        ]

    focus = extract_definition_focus(q)
    if focus:
        return "definition", [focus]

    return "general", []


def greeting_consistent(answer: str) -> bool:
    a = answer.lower()
    greeting_terms = (
        "こんにちは", "こんばんは", "おはよう", "お疲れ", "はじめまして",
        "よろしく", "話しましょう", "何について", "hello", "hi",
    )
    return any(x in a for x in greeting_terms)



def _normalize_definition_surface(text: str) -> str:
    norm = _normalize_for_similarity(text)
    # Strip sentence-final punctuation and light copula/polite endings so that
    # "人工知能", "人工知能です", and "人工知能です。" are treated alike.
    norm = re.sub(r"[。．.!！?？、,]+$", "", norm)
    for ending in ("です", "だ", "である", "です。", "だ。", "である。"):
        e = _normalize_for_similarity(ending)
        if norm.endswith(e):
            norm = norm[:-len(e)]
            break
    return norm


def definition_is_echo_only(slot: str, answer: str) -> bool:
    answer_norm = _normalize_definition_surface(answer)
    aliases = {
        _normalize_definition_surface(alias)
        for alias in strict_synonyms(slot)
    }
    if answer_norm in aliases:
        return True

    # Also reject simple copular synonym statements such as
    # "人工知能はAIです" when asked for a definition.
    compact = _normalize_for_similarity(answer)
    for left in strict_synonyms(slot):
        left_n = _normalize_for_similarity(left)
        for right in strict_synonyms(slot):
            right_n = _normalize_definition_surface(right)
            if left_n == right_n:
                continue
            candidates = (
                f"{left_n}は{right_n}です",
                f"{left_n}は{right_n}",
            )
            if _normalize_definition_surface(compact) in {
                _normalize_definition_surface(x) for x in candidates
            }:
                return True
    return False


def definition_is_uninformative(slot: str, answer: str) -> bool:
    if definition_is_echo_only(slot, answer):
        return True

    a = _normalize_for_similarity(answer)
    # Reject vacuous templates that mention the focus but provide no category,
    # function, property, relation, or differentiating content.
    vacuous_patterns = (
        "同じ種類のもの",
        "同じもの",
        "ものです",
        "ものだ",
        "ものです。",
        "ものだ。",
        "一種です",
        "一種だ",
        "何かです",
        "何かだ",
    )
    if any(_normalize_for_similarity(p) in a for p in vacuous_patterns):
        # "Xは言語モデルです" should pass because it supplies a concrete category.
        informative_terms = (
            "モデル", "装置", "技術", "方式", "手法", "プロセッサ",
            "処理装置", "ソフトウェア", "ハードウェア", "システム",
            "人工知能", "言語モデル", "大規模言語モデル",
        )
        if not any(_normalize_for_similarity(t) in a for t in informative_terms):
            return True

    # Require at least some content beyond the focus/alias and light particles.
    # A concrete category relation is informative even when the answer is short:
    # "LLMは言語モデルです", "GPUは画像処理装置です".
    if any(
        _normalize_for_similarity(category) in a
        for category in definition_categories(slot)
    ):
        return False

    stripped = a
    for alias in sorted(strict_synonyms(slot), key=len, reverse=True):
        stripped = stripped.replace(_normalize_for_similarity(alias), "")
    stripped = re.sub(r"(とは|は|って|を|が|に|で|の|です|だ|である|。|、|,|\.)", "", stripped)
    return len(stripped) < 2


def slot_coverage_check(
    intent: str,
    slots: list[str],
    answer: str,
) -> tuple[bool, float, str]:
    if intent == "greeting":
        if greeting_consistent(answer):
            return True, 1.0, "greeting matched"
        return False, 0.0, "greeting intent mismatch"

    if not slots:
        return True, 1.0, "no required slots"

    hits = [slot for slot in slots if slot_present(slot, answer)]
    coverage = len(hits) / len(slots)

    if intent == "definition":
        if coverage < 1.0:
            return False, coverage, f"definition focus missing: {slots[0]}"

        if definition_is_echo_only(slots[0], answer):
            return False, coverage, "definition answer is only an echo"
        if definition_is_uninformative(slots[0], answer):
            return False, coverage, "definition answer is uninformative"

        return True, coverage, "definition slot covered"

    if intent == "comparison":
        if coverage < 1.0:
            missing = [s for s in slots if not slot_present(s, answer)]
            return (
                False,
                coverage,
                "comparison slot missing: " + ", ".join(missing),
            )

        comparison_cues = (
            "一方", "対して", "違", "比べ", "比較", "対照", "より",
            "得意", "汎用", "並列", "制御",
        )
        if not any(cue in answer for cue in comparison_cues):
            return False, coverage, "comparison relation missing"

        return True, coverage, "comparison slots covered"

    return True, coverage, "slots covered"


def focus_consistent(question: str, answer: str) -> tuple[bool, str]:
    focus = extract_definition_focus(question)
    if not focus:
        return True, "no explicit focus"

    if focus.lower() in answer.lower():
        return True, "focus preserved"

    # Strong mismatch for acronym/technical-term definition questions.
    if re.fullmatch(r"[A-Za-z0-9_+.#\-]{2,20}", focus):
        return False, f"definition focus missing: {focus}"

    return True, "focus not mandatory"



ANSWER_CONCEPT_RULES = {
    "ai": {
        "required_any": ("人工知能", "知的", "技術", "システム", "llm"),
        "forbidden": (),
    },
    "人工知能": {
        "required_any": ("人工知能", "知的", "技術", "システム", "ai"),
        "forbidden": (),
    },
    "cpu": {
        "required_any": ("中央処理装置", "プロセッサ", "命令", "制御", "演算"),
        "forbidden": ("人工知能", "llm", "言語モデル"),
    },
    "gpu": {
        "required_any": ("gpu", "並列", "演算", "プロセッサ", "画像処理装置"),
        "forbidden": ("人工知能", "llm", "言語モデル"),
    },
    "llm": {
        "required_any": ("llm", "言語モデル", "大規模言語モデル", "言語"),
        "forbidden": ("画像処理装置", "中央処理装置"),
    },
}


def answer_concept_consistency(
    question: str,
    answer: str,
) -> tuple[bool, str]:
    q = question.lower()
    matched_key = None
    for key in ("gpu", "cpu", "llm", "人工知能", "ai"):
        if key in q:
            matched_key = key
            break

    if matched_key is None:
        return True, "no answer concept rule"

    rule = ANSWER_CONCEPT_RULES[matched_key]
    a = answer.lower()

    required = rule["required_any"]
    if required and not any(term.lower() in a for term in required):
        return False, f"answer concept missing for {matched_key.upper()}"

    bad = [term for term in rule["forbidden"] if term.lower() in a]
    if bad:
        return False, (
            f"answer concept conflict for {matched_key.upper()}: "
            + ", ".join(bad)
        )

    return True, "answer concept consistent"

CANONICAL_DEFINITIONS = {
    "cpu": "CPUは、命令を解釈して演算や制御を実行する中央処理装置である。",
    "gpu": "GPUは、多数の演算を並列に実行することを得意とする処理装置である。",
    "cuda": "CUDAは、NVIDIAのGPUを汎用計算に利用するための並列計算基盤である。",
}


def canonical_definition_lookup(question: str) -> tuple[str, str] | None:
    """Return a stable canonical definition for explicitly validated concepts."""
    focus = extract_concept_query_focus(question)
    if not focus:
        focus = extract_bare_concept_focus(question)
    if not focus:
        return None

    answer = CANONICAL_DEFINITIONS.get(focus.lower())
    if not answer:
        return None
    return focus, answer


KNOWN_QUERY_CONCEPTS = {
    "ai", "人工知能",
    "llm", "大規模言語モデル",
    "cpu", "gpu", "cuda",
    "コンピュータ",
    "量子力学", "量子コンピュータ",
    "semantic", "セマンティック", "セマンティックデータ",
    # Persona/identity concepts that are explicitly trained in the Nagato SFT.
    "長門", "長門有希",
}


def extract_bare_concept_focus(question: str) -> str:
    """Extract a conservative bare concept token such as 'CUDA' or '宇宙'.

    This intentionally excludes conversational/persona phrases and malformed
    punctuation. It is used only as a pre-generation safety gate.
    """
    q = question.strip()
    if not q:
        return ""

    # Exclude whitespace, sentence punctuation, particles and command-like text.
    if re.search(r"[\s。、！？!?？,:：;；]", q):
        return ""
    if len(q) > 24:
        return ""

    # Technical identifiers or short Japanese noun-like tokens.
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+.#\-]{1,23}", q):
        return q
    if re.fullmatch(r"[一-龯々ァ-ヶー]{2,24}", q):
        return q

    return ""


BARE_DEFINITION_CONCEPTS = {
    "ai", "人工知能",
    "llm", "大規模言語モデル",
    "cpu", "gpu", "cuda",
    "コンピュータ",
    "量子力学", "量子コンピュータ",
    "semantic", "セマンティック", "セマンティックデータ",
}


def canonicalize_bare_known_query(question: str) -> str:
    """Turn a validated bare knowledge concept into an explicit definition query.

    Example: "CPU" -> "CPUとは".

    Persona nouns such as "長門" are intentionally excluded; this helper is
    only for concepts whose bare form semantically means "tell me what X is".
    """
    focus = extract_bare_concept_focus(question)
    if not focus:
        return question
    if focus.lower() not in BARE_DEFINITION_CONCEPTS:
        return question
    return f"{focus}とは"


def extract_concept_query_focus(question: str) -> str:
    """Extract a concept from the explicit query forms used by the pre-gate."""
    q = question.strip()
    patterns = (
        r"^(.+?)(?:とは)$",
        r"^(.+?)(?:って何)$",
        r"^(.+?)(?:について教えて)$",
        r"^(.+?)(?:を説明して)$",
        r"^(.+?)(?:を簡単に説明して)$",
        r"^([^\s。、！？?]{1,24})は$",
    )
    for pattern in patterns:
        m = re.fullmatch(pattern, q)
        if m:
            return m.group(1).strip()
    return ""


def raw_corpus_knows_focus(
    focus: str,
    corpus_path: str | Path = DEFAULT_RAW_KNOWLEDGE_CORPUS,
    min_occurrences: int = 2,
) -> bool:
    """Return True when a concept is materially present in the local raw corpus.

    This only relaxes the *pre-generation* lexical gate.  The generated answer
    must still pass the normal confidence/semantic gate.
    """
    path = resolve_runtime_path(corpus_path)
    if not path.exists() or not focus:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return text.count(focus) >= max(1, min_occurrences)


def semantic_proposition_lookup(
    question: str,
    proposition_path: str | Path = DEFAULT_PROPOSITION_STORE,
) -> tuple[str, str] | None:
    """Return a composed answer from atomic propositions for concept queries."""
    focus = extract_concept_query_focus(question)
    if not focus:
        focus = extract_bare_concept_focus(question)
    if not focus:
        return None

    path = resolve_runtime_path(proposition_path)
    if not path.exists():
        return None

    answer = compose_semantic_proposition_subject(path, focus)
    if not answer:
        return None
    return focus, answer


def semantic_knowledge_lookup(
    question: str,
    knowledge_path: str | Path = DEFAULT_SEMANTIC_KNOWLEDGE,
) -> tuple[str, str] | None:
    """Look up a merged semantic answer for an explicit concept query.

    Returns (concept, answer) when the concept is present in the semantic
    proposition knowledge file.  This is retrieval, not generation.
    """
    focus = extract_concept_query_focus(question)
    if not focus:
        focus = extract_bare_concept_focus(question)
    if not focus:
        return None

    path = resolve_runtime_path(knowledge_path)
    if not path.exists():
        return None

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for raw in lines:
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue

        concept = str(row.get("concept", "")).strip()
        answer = str(row.get("assistant", "")).strip()
        if concept == focus and answer:
            return concept, answer

    return None


def pre_generation_unknown_concept(
    question: str,
    promoted_concepts: set[str] | None = None,
) -> tuple[bool, str]:
    """
    Question-side lexical concept gate for definition/explanation prompts.

    Returns (is_unknown, focus). This is intentionally conservative and only
    triggers on explicit concept-query forms so ordinary persona/chat prompts
    such as '本は好きですか' are unaffected.
    """
    explicit_focus = extract_concept_query_focus(question)
    bare_focus = ""
    focus = explicit_focus
    if not focus:
        bare_focus = extract_bare_concept_focus(question)
        focus = bare_focus
    if not focus:
        return False, ""

    norm = focus.lower()

    # Pronouns / conversational identity prompts are not knowledge concepts.
    # They must reach the normal persona/chat path instead of being rejected
    # by the lexical concept pre-gate.
    NON_CONCEPT_FOCI = {
        "あなた", "貴方", "きみ", "君", "おまえ", "お前",
        "わたし", "私", "ぼく", "僕",
    }
    if focus in NON_CONCEPT_FOCI:
        return False, focus

    if norm in KNOWN_QUERY_CONCEPTS:
        return False, focus
    if promoted_concepts and norm in promoted_concepts:
        return False, focus

    # v10.9.0 semantic integration: a bare noun such as "時間" must not
    # become KNOWN merely because the raw corpus contains the token.  Bare
    # concept queries require validated/promoted knowledge or a retrieval hit
    # handled before this gate.  This prevents corpus-frequency hallucination.
    if bare_focus:
        return True, focus

    # Preserve the legacy raw-corpus relaxation only for explicit concept
    # queries such as "Xとは".  Generated answers still pass downstream gates.
    if raw_corpus_knows_focus(focus):
        return False, focus
    return True, focus


def classify_resolution(
    accepted: bool,
    reason: str,
    question: str = "",
) -> tuple[str, str]:
    if accepted:
        return "ACCEPT", "none"

    q = question.strip()

    input_markers = (
        "malformed repeated intent",
        "malformed technical-term suffix",
        "incoherent multi-concept input",
        "malformed short technical query",
        "empty input",
        "missing query subject",
    )
    if any(marker in reason for marker in input_markers):
        return "INPUT_REJECT", "ask/rephrase"

    # Unknown named entities are not automatically a local training failure.
    # They should be routed to external knowledge or explicit teaching.
    if "unknown entity missing" in reason:
        return "UNKNOWN_KNOWLEDGE", "retrieve/teach"

    # Known concepts with incomplete or contradictory answers indicate that
    # the current model should be taught rather than the gate relaxed.
    learning_markers = (
        "definition answer is only an echo",
        "definition answer is uninformative",
        "concept slot missing",
        "answer concept missing",
        "answer concept conflict",
    )
    if any(marker in reason for marker in learning_markers):
        return "LEARNING_GAP", "teach/train"

    # A definition-like request for an unknown focus term is also knowledge
    # absence even when the generation failed before entity matching.
    focus = extract_definition_focus(q)
    known_focuses = {"ai", "人工知能", "cpu", "gpu", "llm", "大規模言語モデル", "cuda"}
    if focus and focus.lower() not in known_focuses:
        return "UNKNOWN_KNOWLEDGE", "retrieve/teach"

    # Confidence/agreement/quality failures are routed for gate review.
    return "GATE_REVIEW", "review/gate"


def semantic_consistency_check(
    model: LanguageModel,
    tokenizer: Tokenizer,
    current_question: str,
    answer: str,
    history: List[Tuple[str, str]],
    contamination_margin: float,
) -> tuple[bool, float, float, str, str, list[str], float]:
    intent, slots = classify_intent_and_slots(current_question)

    suspicious_entities = extract_suspicious_entities(current_question)
    missing_entities = [
        entity
        for entity in suspicious_entities
        if entity.lower() not in answer.lower()
    ]
    if missing_entities:
        return (
            False, 0.0, 0.0,
            "unknown entity missing: " + ", ".join(missing_entities),
            intent, slots, 0.0,
        )

    # Compute semantic similarity before lexical slot rejection so that
    # canonical concepts can use a conservative semantic fallback.
    qv = semantic_vector(model, tokenizer, current_question)
    av = semantic_vector(model, tokenizer, answer)
    current_sim = float(torch.dot(qv, av).item())

    slots_ok, slot_coverage, slot_reason = slot_coverage_check(
        intent, slots, answer
    )

    # v1.6.4 hard constraint: a definition that is only the focus term or
    # one of its aliases is incomplete. Do not let concept fallback override it.
    if intent == "definition" and slots:
        if definition_is_echo_only(slots[0], answer):
            return (
                False,
                current_sim,
                0.0,
                "definition answer is only an echo",
                intent,
                slots,
                slot_coverage,
            )
        if definition_is_uninformative(slots[0], answer):
            return (
                False,
                current_sim,
                0.0,
                "definition answer is uninformative",
                intent,
                slots,
                slot_coverage,
            )

    if not slots_ok and intent in ("definition", "comparison") and slots:
        covered = 0
        evidence: list[str] = []
        for slot in slots:
            if slot_present(slot, answer):
                covered += 1
                evidence.append(f"{slot}:lexical")
                continue

            categories = definition_categories(slot)
            answer_norm = _normalize_for_similarity(answer)
            matched_category = next(
                (
                    category
                    for category in categories
                    if _normalize_for_similarity(category) in answer_norm
                ),
                None,
            )
            if intent == "definition" and matched_category is not None:
                covered += 1
                evidence.append(
                    f"{slot}:category({matched_category})"
                )
                continue

            concept_ok, concept_sim, concept_margin = concept_slot_match(
                model=model,
                tokenizer=tokenizer,
                slot=slot,
                answer=answer,
            )
            if concept_ok:
                covered += 1
                evidence.append(
                    f"{slot}:concept({concept_sim:.3f}/{concept_margin:.3f})"
                )
            else:
                evidence.append(
                    f"{slot}:miss({concept_sim:.3f}/{concept_margin:.3f})"
                )

        slot_coverage = covered / len(slots)
        if slot_coverage >= 1.0:
            slots_ok = True
            slot_reason = "concept-aware slots: " + ", ".join(evidence)
        else:
            return (
                False,
                current_sim,
                0.0,
                "concept slot missing: " + ", ".join(evidence),
                intent,
                slots,
                slot_coverage,
            )

    # Greeting turns are intentionally short and semantically broad.
    # Slot/intent matching is more reliable than history similarity here.
    # A non-greeting answer must not pass merely because generation confidence
    # or probe agreement is high.
    if intent == "greeting":
        return (
            slots_ok, current_sim, -1.0, slot_reason,
            intent, slots, slot_coverage,
        )

    previous_sims: List[float] = []
    contamination_candidates: List[float] = []
    for old_user, _ in history[-3:]:
        pv = semantic_vector(model, tokenizer, old_user)
        old_answer_sim = float(torch.dot(pv, av).item())
        old_current_sim = float(torch.dot(pv, qv).item())
        previous_sims.append(old_answer_sim)

        # v1.6.3: history is suspicious only when the previous question is
        # materially different from the current question. Closely related
        # paraphrases/synonyms must not be rejected merely because the answer
        # also matches the previous turn.
        if old_current_sim < 0.82:
            contamination_candidates.append(old_answer_sim)

    previous_best = max(previous_sims) if previous_sims else -1.0
    contamination_best = (
        max(contamination_candidates)
        if contamination_candidates
        else -1.0
    )

    if (
        contamination_candidates
        and contamination_best > current_sim + contamination_margin
    ):
        return (
            False,
            current_sim,
            previous_best,
            "history contamination",
            intent,
            slots,
            slot_coverage,
        )

    return (
        True,
        current_sim,
        previous_best,
        slot_reason,
        intent,
        slots,
        slot_coverage,
    )


def calibrated_agreement_threshold(
    intent: str,
    semantic_ok: bool,
    slot_coverage: float,
    base_threshold: float,
) -> float:
    """Relax agreement only when semantic structure is already verified."""
    if (
        semantic_ok
        and slot_coverage >= 1.0
        and intent in ("definition", "comparison")
    ):
        return min(base_threshold, 0.30)
    return base_threshold


def evaluate_unknown_gate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    results: List[GenerationResult],
    min_confidence: float,
    min_token_confidence: float,
    min_mean_margin: float,
    min_agreement: float,
    min_semantic_agreement: float,
    allow_semantic_rescue: bool,
) -> Tuple[bool, float, float, float, str]:
    if not results:
        return False, 0.0, 0.0, 0.0, "no generation"

    primary = results[0]

    if malformed_or_unstable(primary.text):
        return (
            False,
            primary.mean_confidence,
            0.0,
            0.0,
            "malformed/repetitive output",
        )

    # v1.6.8 Output Quality Gate: semantic agreement cannot rescue a response
    # containing a locally very unlikely token sequence.
    if primary.min_confidence < min_token_confidence:
        return (
            False,
            primary.mean_confidence,
            0.0,
            0.0,
            "low local token confidence",
        )

    if primary.mean_top2_margin < min_mean_margin:
        return (
            False,
            primary.mean_confidence,
            0.0,
            0.0,
            "low generation margin",
        )

    lexical_agreement = mean_pairwise_agreement(results)
    semantic_agreement = mean_pairwise_semantic_agreement(
        model,
        tokenizer,
        results,
    )

    if primary.mean_confidence < min_confidence:
        return (
            False,
            primary.mean_confidence,
            lexical_agreement,
            semantic_agreement,
            "low token confidence",
        )

    if lexical_agreement >= min_agreement:
        return (
            True,
            primary.mean_confidence,
            lexical_agreement,
            semantic_agreement,
            "accepted",
        )

    # v1.6.7 semantic rescue is deliberately conservative: it is enabled only
    # after the primary answer has passed semantic/slot validation.
    if allow_semantic_rescue and semantic_agreement >= min_semantic_agreement:
        return (
            True,
            primary.mean_confidence,
            lexical_agreement,
            semantic_agreement,
            "semantic probe agreement",
        )

    return (
        False,
        primary.mean_confidence,
        lexical_agreement,
        semantic_agreement,
        "low response agreement",
    )


def print_info(
    model: LanguageModel,
    tokenizer: Tokenizer,
    checkpoint: dict,
    device: torch.device,
    model_path: Path,
    tokenizer_path: Path,
    args: argparse.Namespace,
) -> None:
    print()
    print("==============================================")
    print(" LLM_TRY Chat - v10.11.8 Composite Teacher Fidelity Gate")
    print("==============================================")
    print("Device          :", device)
    if device.type == "cuda":
        print("GPU             :", torch.cuda.get_device_name(0))
    print("Model           :", model_path)
    print("Tokenizer       :", tokenizer_path)
    print("Vocabulary size :", tokenizer.vocab_size)
    print("Parameters      :", f"{model.parameter_count:,}")
    print("Context length  :", model.context_length)
    print("d_model         :", model.d_model)
    print("Layers          :", model.num_layers)
    print("Attention heads :", model.num_heads)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint loss :", checkpoint.get("loss"))
    print("Unknown reject  :", args.unknown_rejection)
    if args.unknown_rejection:
        print("Probe count     :", args.probe_count)
        print("Min confidence  :", args.min_confidence)
        print("Min token conf  :", args.min_token_confidence)
        print("Min mean margin :", args.min_mean_margin)
        print("Min agreement   :", args.min_agreement)
        print("Min sem agree   :", args.min_semantic_agreement)
        print("Min int semantic:", args.min_internalized_fidelity)
        print(
            "Min int lexical :",
            args.min_internalized_lexical_coverage,
        )
        print(
            "Min int required:",
            args.min_internalized_required_coverage,
        )
        print("Fallback        :", UNKNOWN_REPLY)
        print("Context policy  : minimal")
        print("Teaching queue  :", args.teaching_queue)
        print("Knowledge queue :", args.knowledge_queue)
        print("Gate review q   :", args.gate_review_queue)
        print("Proposition db  :", args.propositions)
        print("Subject index   :", args.subject_index)
        print("Typed index     :", args.typed_subject_index)
        print(
            "Concept calib   :",
            CALIBRATION_INFO.get("version", "raw fallback")
            if CALIBRATION_INFO.get("loaded")
            else "raw fallback",
        )
        if CALIBRATION_INFO.get("loaded"):
            print("Centroid source : TRAIN ONLY")
            print("Holdout in fit  :", CALIBRATION_INFO.get("holdout_used_for_centroid"))
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        bound = metadata.get("trained_fingerprints", [])
        binding_version = metadata.get("knowledge_binding_version", "")
        print(
            "Checkpoint bind :",
            f"{len(bound) if isinstance(bound, list) else 0} fingerprint(s)",
        )
        if binding_version:
            print("Binding version :", binding_version)
    print()


def checkpoint_trained_fingerprints(
    checkpoint: dict,
) -> frozenset[str]:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        return frozenset()
    values = metadata.get("trained_fingerprints", [])
    if not isinstance(values, (list, tuple, set)):
        return frozenset()
    return frozenset(str(value) for value in values)


def main() -> None:
    args = parse_args()

    tokenizer_path = resolve_runtime_path(args.tokenizer)
    model_path = choose_startup_model(
        args.model,
        online_output=args.online_output,
        learning_state=args.learning_state,
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}\n"
            "Place tokenizer-v0.7-bpe.json under LLM_TRY/model/ or "
            "keep the sibling LLM_GPU/model/ directory available. "
            "You can also pass --tokenizer <path>."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Chat model not found: {model_path}\n"
            "Expected the LLM_TRY v9.4 Nagato SFT checkpoint.\n"
            "Place it under LLM_TRY/model/ or pass --model <path>."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = Tokenizer.load(str(tokenizer_path))
    model, checkpoint = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )

    calibration_path = resolve_runtime_path(args.concept_calibration)
    load_concept_calibration(calibration_path, device)

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"{tokenizer.vocab_size} != {model.vocab_size}"
        )

    print_info(
        model=model,
        tokenizer=tokenizer,
        checkpoint=checkpoint,
        device=device,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        args=args,
    )

    print("Commands:")
    print("  /reset        clear conversation history")
    print("  /info         show model/checkpoint information")
    print("  /learn on     capture accepted turns for later training")
    print("  /learn off    stop capture")
    print("  /learn status show capture state and saved-pair count")
    print("  /teach TEXT   save a corrected answer for the previous user turn")
    print("  /teachq Q => A teach an explicit question/answer pair")
    print("  /good         approve and save the previous AI answer for learning")
    print("  /train        run incremental training and reload checkpoint")
    print("  /maintain     recover the previous question from trusted teaching data")
    print("  /maintain all recover all pending trusted teaching candidates")
    print("  /propteach S  decompose and persist semantic proposition statement")
    print("  /prop X       compose stored propositions for subject X")
    print("  /props        list atomic semantic propositions")
    print("  /propsync     sync all atomic propositions to unified semantic memory")
    print("  /subject X    show X => X+predicate mappings")
    print("  /subjects     list subject-keyed proposition index")
    print("  /typedsubject X show X => predicate type => statement")
    print("  /typedsubjects list typed subject proposition index")
    print("  /internalized list concepts proven consumed by /train")
    print("  /kstate Q     inspect resolved knowledge state for query Q")
    print("  /dispatch Q   inspect dispatcher action for query Q")
    print("  /provenance Q inspect source/origin evidence for query Q")
    print("  /truth X      inspect effective truth state for concept X")
    print("  /truths       list explicit truth-state records")
    print("  /truthset X STATE [=> CORRECTION]")
    print("  /semstatus    show Semantic Knowledge Architecture status")
    print("  /semantic X   inspect all semantic knowledge layers for X")
    print("  /promotions   list pending UNKNOWN_KNOWLEDGE requests")
    print("  /promotionstatus show pending/promoted/verified lifecycle counts")
    print("  /promote X => STATEMENT  promote pending concept to semantic knowledge")
    print("  /exit         quit")
    print()

    history: List[Tuple[str, str]] = []
    learning_log = Path(args.learning_log)
    learning_state = Path(args.learning_state)
    proposition_path = resolve_runtime_path(args.propositions)
    unified_semantic_path = resolve_runtime_path(DEFAULT_SEMANTIC_KNOWLEDGE)
    subject_index_path = resolve_runtime_path(args.subject_index)
    typed_subject_index_path = resolve_runtime_path(
        args.typed_subject_index
    )
    truth_store_path = resolve_runtime_path(args.truth_store)

    semantic_config = SemanticKnowledgeConfig(
        proposition_path=proposition_path,
        subject_index_path=subject_index_path,
        typed_index_path=typed_subject_index_path,
        unified_path=unified_semantic_path,
        learning_log=learning_log,
        learning_state=learning_state,
        raw_corpus_path=resolve_runtime_path(
            DEFAULT_RAW_KNOWLEDGE_CORPUS
        ),
        truth_store_path=truth_store_path,
        canonical_definitions=CANONICAL_DEFINITIONS,
        checkpoint_fingerprints=checkpoint_trained_fingerprints(
            checkpoint
        ),
    )
    semantic_knowledge = SemanticKnowledgeArchitecture(
        semantic_config
    )
    migrated_legacy_queue = consolidate_legacy_resolved(
        Path(args.knowledge_queue)
    )
    if migrated_legacy_queue:
        print(
            f"[knowledge queue lifecycle: migrated "
            f"{migrated_legacy_queue} legacy resolved row(s)]"
        )
        print()

    baseline_count = initialize_learning_state_if_missing(
        learning_state,
        learning_log,
        assume_existing_trained=(
            model_path != Path(DEFAULT_MODEL)
        ),
    )
    if baseline_count:
        print(
            f"[v1.6.2 baseline initialized: "
            f"{baseline_count} existing trusted pair(s) marked as trained]"
        )
        print()
    learning_enabled = bool(args.learn)
    last_user_text: str | None = None
    last_ai_reply: str | None = None

    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue

        normalized_user_text = normalize_runtime_input(user_text)
        if normalized_user_text and normalized_user_text != user_text:
            print(f"[normalized input: {user_text!r} -> {normalized_user_text!r}]")
            user_text = normalized_user_text
        if not user_text:
            continue

        command = user_text.lower()

        if command in ("/exit", "exit", "quit"):
            break

        if command == "/reset":
            history.clear()
            print("[conversation history cleared]")
            print()
            continue

        if command == "/info":
            print_info(
                model=model,
                tokenizer=tokenizer,
                checkpoint=checkpoint,
                device=device,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                args=args,
            )
            print("Learning capture :", learning_enabled)
            print("Learning log     :", learning_log)
            print("Saved pairs      :", learning_log_count(learning_log))
            print()
            continue

        if command == "/learn on":
            learning_enabled = True
            print("[learning capture enabled]")
            print()
            continue

        if command == "/learn off":
            learning_enabled = False
            print("[learning capture disabled]")
            print()
            continue

        if command == "/learn status":
            print(
                f"[learning={'ON' if learning_enabled else 'OFF'}, "
                f"pairs={learning_log_count(learning_log)}, "
                f"log={learning_log}]"
            )
            print()
            continue

        if command == "/props":
            propositions = load_semantic_propositions(proposition_path)
            if not propositions:
                print("[semantic propositions: empty]")
            else:
                print(
                    f"[semantic propositions: {len(propositions)} atomic item(s)]"
                )
                for index, item in enumerate(propositions, 1):
                    print(
                        f"  {index:02d}. "
                        f"subject={item.subject!r} value={item.value!r}"
                    )
            print()
            continue

        if command.startswith("/propteach "):
            statement = user_text[len("/propteach "):].strip()
            added = semantic_knowledge.teach_proposition(statement)
            if not added:
                print(
                    "[proposition teaching rejected: expected "
                    "XはYである or Xは、Yであり、Zである]"
                )
            else:
                print(
                    f"[semantic knowledge teach: {len(added)} atomic item(s)]"
                )
                for item in added:
                    print(
                        f"  + subject={item.subject!r} value={item.value!r}"
                    )
                rendered = compose_semantic_proposition_subject(
                    proposition_path,
                    added[0].subject,
                )
                print(f"[proposition composed: {rendered}]")
                print(
                    "[semantic layers synchronized: "
                    "proposition -> subject -> typed -> unified]"
                )
            print()
            continue

        if command == "/propsync":
            sync = semantic_knowledge.sync_all()
            print(
                f"[semantic knowledge sync: "
                f"atomic={sync.atomic_count}, "
                f"subject_index={sync.subject_index_count}, "
                f"typed_index={sync.typed_index_count}, "
                f"unified_subjects={sync.unified_subject_count}]"
            )
            print()
            continue

        if command == "/promotionstatus":
            summary = lifecycle_summary(
                Path(args.knowledge_queue)
            )
            print(
                f"[knowledge queue lifecycle: "
                f"pending={summary.pending}, "
                f"promoted={summary.promoted}, "
                f"verified={summary.verified}, "
                f"legacy_resolved={summary.legacy_resolved}, "
                f"other={summary.other}, "
                f"total={summary.total}]"
            )
            print()
            continue

        if command == "/promotions":
            pending = pending_knowledge_requests(
                Path(args.knowledge_queue)
            )
            if not pending:
                print("[knowledge promotions: no pending UNKNOWN_KNOWLEDGE requests]")
            else:
                print(
                    f"[knowledge promotions: {len(pending)} pending request(s)]"
                )
                for index, row in enumerate(pending, 1):
                    print(
                        f"  {index:02d}. user={row.get('user', '')!r}, "
                        f"reason={row.get('reason', '')!r}, "
                        f"timestamp={row.get('timestamp', '')!r}"
                    )
            print()
            continue

        if command.startswith("/promote "):
            payload = user_text[len("/promote "):].strip()
            if "=>" not in payload:
                print("[usage: /promote CONCEPT => STATEMENT]")
                print()
                continue

            concept, statement = [
                part.strip()
                for part in payload.split("=>", 1)
            ]
            if not concept or not statement:
                print("[usage: /promote CONCEPT => STATEMENT]")
                print()
                continue

            promotion = promote_knowledge(
                semantic_knowledge,
                Path(args.knowledge_queue),
                concept,
                statement,
            )
            if not promotion.promoted:
                print(
                    f"[knowledge promotion rejected: "
                    f"concept={concept!r}, reason={promotion.reason}]"
                )
            else:
                post = promotion.post_result
                truth_state = (
                    post.truth_state
                    if post is not None
                    else ""
                )
                print(
                    f"[knowledge promoted: concept={concept!r}, "
                    f"atomic={len(promotion.propositions)}, "
                    f"queue_promoted={promotion.queue_promoted}, "
                    f"post_state={post.state if post else '-'}, "
                    f"post_action={post.action if post else '-'}, "
                    f"truth_state={truth_state or '-'}]"
                )
                for item in promotion.propositions:
                    print(
                        f"  + {item.subject} => {item.value}"
                    )
                print(
                    "[promotion pipeline: UNKNOWN -> validated proposition "
                    "-> subject/typed/unified sync -> bare routing enabled]"
                )
            print()
            continue

        if command.startswith("/semantic "):
            concept = user_text[len("/semantic "):].strip()
            if not concept:
                print("[usage: /semantic CONCEPT]")
            else:
                snapshot = snapshot_semantic_knowledge(
                    semantic_knowledge,
                    concept,
                )
                result = snapshot.result
                state = result.knowledge_state
                dispatch = result.dispatch
                truth = result.truth
                truth_result = result.truth_result
                provenance = result.provenance

                print(
                    f"[semantic snapshot: concept={snapshot.concept!r}, "
                    f"query={snapshot.query!r}]"
                )

                if snapshot.atomic_propositions:
                    print(
                        f"  Proposition  : {snapshot.proposition_count} atomic item(s)"
                    )
                    for item in snapshot.atomic_propositions:
                        print(
                            f"    - {item.subject} => {item.value}"
                        )
                else:
                    print("  Proposition  : none")

                print(
                    "  Subject Index : "
                    + (snapshot.subject_answer or "none")
                )

                if snapshot.typed_rows:
                    print(
                        f"  Typed Index   : {snapshot.typed_count} item(s)"
                    )
                    for row in snapshot.typed_rows:
                        print(
                            f"    - {row.predicate_type}: {row.statement}"
                        )
                else:
                    print("  Typed Index   : none")

                if snapshot.unified_row is not None:
                    print(
                        "  Unified Memory: "
                        + str(snapshot.unified_row.get("assistant", ""))
                    )
                    print(
                        f"    source={snapshot.unified_row.get('source', '')!r}, "
                        f"updated_at={snapshot.unified_row.get('updated_at', '')!r}"
                    )
                else:
                    print("  Unified Memory: none")

                if snapshot.internalized is not None:
                    internalized = snapshot.internalized
                    print(
                        f"  Internalized : yes, "
                        f"trained_pairs={internalized.trained_pairs}, "
                        f"sources={','.join(internalized.sources)}"
                    )
                    print(
                        f"    latest_question={internalized.latest_question!r}"
                    )
                else:
                    print("  Internalized : no")

                if provenance is not None:
                    print(
                        f"  Provenance   : {provenance.compact()}"
                    )
                else:
                    print("  Provenance   : none")

                if truth is not None:
                    runtime_status = (
                        truth_result.runtime_status
                        if truth_result is not None
                        else "-"
                    )
                    warning = (
                        truth_result.warning
                        if truth_result is not None
                        else ""
                    )
                    print(
                        f"  Truth State  : {truth.state}, "
                        f"runtime={runtime_status}, warning={warning!r}"
                    )
                    if truth.correction:
                        print(
                            f"    correction={truth.correction}"
                        )
                else:
                    print("  Truth State  : none")

                print(
                    f"  Knowledge    : state={state.state}, "
                    f"reason={state.reason}"
                )
                print(
                    f"  Final Route  : action={dispatch.action}, "
                    f"route={dispatch.route}, "
                    f"reason={dispatch.reason}"
                )
                if dispatch.answer:
                    print(f"  Final Answer : {dispatch.answer}")
            print()
            continue

        if command == "/semstatus":
            status = semantic_knowledge.status()
            print(
                f"[semantic knowledge architecture: "
                f"version={status['version']}, "
                f"layers={','.join(status['layers'])}]"
            )
            print(f"  proposition : {status['proposition_path']}")
            print(f"  subject     : {status['subject_index_path']}")
            print(f"  typed       : {status['typed_index_path']}")
            print(f"  unified     : {status['unified_path']}")
            print(f"  internalized: {status['learning_log']}")
            print(f"  truth       : {status['truth_store_path']}")
            print()
            continue

        if command == "/truths":
            records = load_truth_records(truth_store_path)
            if not records:
                print("[truth states: empty; unresolved concepts default to UNVERIFIED]")
            else:
                print(f"[truth states: {len(records)} explicit record(s)]")
                for index, record in enumerate(records, 1):
                    correction = (
                        f" correction={record.correction!r}"
                        if record.correction
                        else ""
                    )
                    print(
                        f"  {index:02d}. concept={record.concept!r} "
                        f"state={record.state} source={record.source} "
                        f"updated_at={record.updated_at!r}"
                        f"{correction}"
                    )
            print()
            continue

        if command.startswith("/truthset "):
            payload = user_text[len("/truthset "):].strip()
            left, sep, correction = payload.partition("=>")
            parts = left.strip().split()
            if len(parts) < 2:
                print(
                    "[usage: /truthset CONCEPT "
                    "TRUE|FALSE|UNVERIFIED|CONTESTED|OUTDATED "
                    "[=> CORRECTION]]"
                )
            else:
                concept = parts[0].strip()
                state_name = parts[1].strip().upper()
                if state_name not in TRUTH_STATES:
                    print(
                        "[invalid truth state: "
                        + state_name
                        + "; expected "
                        + "|".join(TRUTH_STATES)
                        + "]"
                    )
                else:
                    record = upsert_truth_record(
                        truth_store_path,
                        concept,
                        state_name,
                        correction=(correction.strip() if sep else ""),
                        source="chat-manual-truth",
                    )
                    print(
                        f"[truth state saved: concept={record.concept!r}, "
                        f"state={record.state}, "
                        f"correction={record.correction!r}, "
                        f"updated_at={record.updated_at!r}]"
                    )
                    if state_name == "TRUE":
                        verified_count = mark_concept_verified(
                            Path(args.knowledge_queue),
                            concept,
                            truth_source="chat-manual-truth",
                        )
                        if verified_count:
                            print(
                                f"[knowledge queue verified: "
                                f"concept={concept!r}, "
                                f"entries={verified_count}]"
                            )
                    else:
                        revoked_count = revoke_concept_verification(
                            Path(args.knowledge_queue),
                            concept,
                            new_truth_state=state_name,
                        )
                        if revoked_count:
                            print(
                                f"[knowledge queue verification revoked: "
                                f"concept={concept!r}, "
                                f"entries={revoked_count}, "
                                f"truth_state={state_name}]"
                            )
            print()
            continue

        if command.startswith("/truth "):
            concept = user_text[len("/truth "):].strip()
            if not concept:
                print("[usage: /truth CONCEPT]")
            else:
                record = effective_truth_record(
                    truth_store_path,
                    concept,
                )
                explicit = (
                    record.source != "truth-default"
                )
                print(
                    f"[truth concept={record.concept!r}, "
                    f"state={record.state}, explicit={explicit}, "
                    f"source={record.source!r}, "
                    f"updated_at={record.updated_at!r}, "
                    f"reason={record.reason!r}, "
                    f"correction={record.correction!r}]"
                )
            print()
            continue

        if command.startswith("/provenance "):
            query = user_text[len("/provenance "):].strip()
            if not query:
                print("[usage: /provenance QUERY]")
            else:
                result = semantic_knowledge.resolve(query)
                state = result.knowledge_state
                provenance = state.provenance
                if provenance is None:
                    print("[provenance: none]")
                else:
                    print(
                        f"[provenance state={state.state}, "
                        f"focus={state.focus!r}, "
                        f"source={provenance.source!r}, "
                        f"origin={provenance.origin!r}, "
                        f"priority={provenance.retrieval_priority}, "
                        f"timestamp={provenance.timestamp!r}, "
                        f"fingerprint={provenance.fingerprint!r}, "
                        f"evidence={provenance.evidence!r}]"
                    )
                    if provenance.metadata:
                        print(
                            "[provenance metadata: "
                            + ", ".join(
                                f"{key}={value}"
                                for key, value in provenance.metadata.items()
                            )
                            + "]"
                        )
            print()
            continue

        if command.startswith("/dispatch "):
            query = user_text[len("/dispatch "):].strip()
            if not query:
                print("[usage: /dispatch QUERY]")
            else:
                result = semantic_knowledge.resolve(query)
                dispatch = result.dispatch
                truth_record = result.truth
                truth_result = result.truth_result
                provenance_text = (
                    dispatch.provenance.compact()
                    if dispatch.provenance is not None
                    else "none"
                )
                truth_text = (
                    truth_record.state
                    if truth_record is not None
                    else "-"
                )
                warning_text = (
                    truth_result.warning
                    if truth_result is not None
                    else ""
                )
                runtime_text = (
                    truth_result.runtime_status
                    if truth_result is not None
                    else "-"
                )
                print(
                    f"[dispatch action={dispatch.action}, "
                    f"state={dispatch.state}, "
                    f"focus={dispatch.focus!r}, "
                    f"predicate_type={dispatch.predicate_type!r}, "
                    f"terminal={dispatch.is_terminal}, "
                    f"route={dispatch.route}, "
                    f"reason={dispatch.reason}, "
                    f"truth_state={truth_text}, "
                    f"truth_runtime={runtime_text}, "
                    f"truth_warning={warning_text!r}, "
                    f"bare_routed={result.bare_concept_routed}, "
                    f"bare_unknown_blocked={result.bare_unknown_blocked}, "
                    f"bare_focus={result.bare_focus!r}, "
                    f"routed_query={result.routed_query!r}, "
                    f"provenance={provenance_text}]"
                )
                if dispatch.answer:
                    print(f"[dispatch answer: {dispatch.answer}]")
            print()
            continue

        if command.startswith("/kstate "):
            query = user_text[len("/kstate "):].strip()
            if not query:
                print("[usage: /kstate QUERY]")
            else:
                result = semantic_knowledge.resolve(query)
                state = result.knowledge_state
                print(
                    f"[knowledge-state={state.state}, "
                    f"focus={state.focus!r}, "
                    f"predicate_type={state.predicate_type!r}, "
                    f"retrieval={state.is_retrieval}, "
                    f"model_generation={state.permits_model_generation}, "
                    f"bare_routed={result.bare_concept_routed}, "
                    f"bare_unknown_blocked={result.bare_unknown_blocked}, "
                    f"bare_focus={result.bare_focus!r}, "
                    f"routed_query={result.routed_query!r}, "
                    f"reason={state.reason}]"
                )
                if state.answer:
                    print(f"[resolved answer: {state.answer}]")
            print()
            continue

        if command == "/internalized":
            concepts = load_internalized_concepts(
                learning_log,
                learning_state,
            )
            if not concepts:
                print("[internalized concepts: empty]")
            else:
                print(
                    f"[internalized concepts: {len(concepts)} concept(s)]"
                )
                checkpoint_bound = (
                    semantic_config.checkpoint_fingerprints
                    or frozenset()
                )
                for index, concept in enumerate(concepts, 1):
                    source_text = ",".join(concept.sources)
                    bound = any(
                        fp in checkpoint_bound
                        for fp in concept.fingerprints
                    )
                    print(
                        f"  {index:02d}. concept={concept.concept!r} "
                        f"trained_pairs={concept.trained_pairs} "
                        f"binding={'CURRENT' if bound else 'STALE'} "
                        f"sources={source_text} "
                        f"latest_question={concept.latest_question!r}"
                    )
            print()
            continue

        if command == "/typedsubjects":
            grouped = typed_index_dict(typed_subject_index_path)
            if not grouped:
                print("[typed subject index: empty]")
            else:
                print(f"[typed subject index: {len(grouped)} subject(s)]")
                for subject, by_type in grouped.items():
                    print(f"  {subject}:")
                    for predicate_type, statements in by_type.items():
                        for statement in statements:
                            print(
                                f"    {subject} => {predicate_type} => "
                                f"{statement}"
                            )
            print()
            continue

        if command.startswith("/typedsubject "):
            subject = user_text[len("/typedsubject "):].strip()
            mappings = typed_subject_mapping(
                typed_subject_index_path,
                subject,
            )
            if not mappings:
                print(
                    f"[no typed subject propositions for {subject!r}]"
                )
            else:
                for mapping in mappings:
                    print(mapping)
            print()
            continue

        if command == "/subjects":
            grouped = subject_index_dict(subject_index_path)
            if not grouped:
                print("[subject index: empty]")
            else:
                print(f"[subject index: {len(grouped)} subject(s)]")
                for subject, statements in grouped.items():
                    print(f"  {subject}:")
                    for statement in statements:
                        print(f"    {subject} => {statement}")
            print()
            continue

        if command.startswith("/subject "):
            subject = user_text[len("/subject "):].strip()
            mappings = subject_mapping(subject_index_path, subject)
            if not mappings:
                print(f"[no subject-keyed propositions for {subject!r}]")
            else:
                for mapping in mappings:
                    print(mapping)
                composed = compose_subject_from_index(
                    subject_index_path,
                    subject,
                )
                if composed:
                    print(f"[composed: {composed}]")
            print()
            continue

        if command.startswith("/prop "):
            subject = user_text[len("/prop "):].strip()
            answer = compose_semantic_proposition_subject(
                proposition_path,
                subject,
            )
            if answer:
                print(f"AI> {answer}")
            else:
                print(f"[no semantic propositions for {subject!r}]")
            print()
            continue

        if command.startswith("/teachq "):
            payload = user_text[len("/teachq "):].strip()
            if "=>" not in payload:
                print("[usage: /teachq QUESTION => ANSWER]")
                print()
                continue

            question, corrected = [
                part.strip() for part in payload.split("=>", 1)
            ]
            if not question or not corrected:
                print("[usage: /teachq QUESTION => ANSWER]")
                print()
                continue

            input_ok, input_reason = input_quality_check(question)
            if not input_ok:
                print(f"[teaching rejected: invalid question: {input_reason}]")
                print()
                continue

            teaching_ok, teaching_reason = validate_teaching_answer(
                question,
                corrected,
            )
            if not teaching_ok:
                print(f"[teaching rejected: {teaching_reason}]")
                print(f"[hint: {teaching_hint(question)}]")
                print()
                continue

            reactivated = mark_pair_for_retraining(
                learning_state,
                question,
                corrected,
            )
            append_learning_pair(
                learning_log,
                question,
                corrected,
                source="chat-manual",
            )
            if reactivated:
                print("[previously trained pair reactivated for retraining]")
            print(
                f"[explicit manual learning pair saved; "
                f"question={question}, "
                f"pairs={learning_log_count(learning_log)}]"
            )
            print()
            continue

        if command.startswith("/teach "):
            corrected = user_text[len("/teach "):].strip()
            if not corrected:
                print("[usage: /teach corrected answer]")
            elif last_user_text is None:
                print("[no previous user turn to teach]")
            else:
                teaching_ok, teaching_reason = validate_teaching_answer(
                    last_user_text,
                    corrected,
                )
                if not teaching_ok:
                    print(f"[teaching rejected: {teaching_reason}]")
                    print(f"[hint: {teaching_hint(last_user_text)}]")
                else:
                    reactivated = mark_pair_for_retraining(
                        learning_state,
                        last_user_text,
                        corrected,
                    )
                    append_learning_pair(
                        learning_log,
                        last_user_text,
                        corrected,
                        source="chat-manual",
                    )
                    if reactivated:
                        print(
                            "[previously trained pair reactivated for retraining]"
                        )
                    print(
                        f"[manual learning pair saved; "
                        f"pairs={learning_log_count(learning_log)}]"
                    )
            print()
            continue

        if command == "/good":
            if last_user_text is None or last_ai_reply is None:
                print("[no previous AI answer to approve]")
            else:
                append_learning_pair(
                    learning_log,
                    last_user_text,
                    last_ai_reply,
                    source="chat-approved",
                )
                print(
                    f"[approved learning pair saved; "
                    f"pairs={learning_log_count(learning_log)}]"
                )
            print()
            continue

        if command in ("/maintain", "/maintain all"):
            target_question = (
                None if command == "/maintain all"
                else last_user_text
            )
            if command == "/maintain" and target_question is None:
                print("[no previous user turn to maintain]")
                print()
                continue

            matched, reactivated, rejected_old, selections = recover_forgotten_pairs_from_queue(
                Path(args.teaching_queue),
                learning_log,
                learning_state,
                target_question=target_question,
            )
            print(
                f"[maintenance matched={matched}, "
                f"reactivated={reactivated}, "
                f"rejected_old_teachers={rejected_old}]"
            )
            for queued_question, teacher_answer, detail in selections:
                if teacher_answer:
                    print(
                        f"[recovery selected: {queued_question} -> "
                        f"{teacher_answer} ({detail})]"
                    )
                else:
                    print(
                        f"[recovery selected: {queued_question} -> NONE "
                        f"({detail})]"
                    )
                    print("[recovery status=NEEDS_TEACHING]")
                    print(f"[hint: {teaching_hint(queued_question)}]")

            if reactivated:
                print("[run /train to relearn reactivated trusted pairs]")
            elif matched:
                print("[matched trusted pairs are already pending or current]")
            elif selections:
                print("[no valid trusted teacher; use /teach before /train]")
            else:
                print("[no pending teaching candidate for maintenance]")
            print()
            continue

        if command == "/train":
            new_model_path = run_online_training(args, model_path)
            if new_model_path is not None:
                model, checkpoint = LanguageModel.load_checkpoint(
                    str(new_model_path),
                    device=device,
                )
                model_path = new_model_path
                print(f"[reloaded trained checkpoint: {model_path}]")
                semantic_config = replace(
                    semantic_config,
                    checkpoint_fingerprints=(
                        checkpoint_trained_fingerprints(checkpoint)
                    ),
                )
                semantic_knowledge = SemanticKnowledgeArchitecture(
                    semantic_config
                )
                print(
                    f"[checkpoint binding refreshed: "
                    f"{len(semantic_config.checkpoint_fingerprints or ())} "
                    "fingerprint(s)]"
                )
                load_concept_calibration(calibration_path, device)
                history.clear()
                last_user_text = None
                last_ai_reply = None
                print("[conversation history cleared after training]")
            print()
            continue

        last_user_text = user_text
        internalized_record = None

        input_ok, input_reason = input_quality_check(user_text)
        if not input_ok:
            print(f"AI> {UNKNOWN_REPLY}")
            if args.show_risk and args.unknown_rejection:
                print(
                    f"[gate=UNKNOWN, confidence=0.000, "
                    f"min_tok_conf=0.000, mean_margin=0.000, "
                    f"agreement=0.000, sem_agreement=0.000, "
                    f"intent=input_quality, slots=-, slot_cov=0.00, "
                    f"qa_sim=0.000, prev_sim=-1.000, "
                    f"agr_th={args.min_agreement:.2f}, "
                    f"context_turns=0, resolution=INPUT_REJECT, "
                    f"action=ask/rephrase, reason={input_reason}]"
                )
            print("[route=request rephrase]")
            print("[0 generated probe tokens, 0.00s, 0.0 tok/s]")
            print()
            last_ai_reply = None
            continue

        semantic_result = semantic_knowledge.resolve(
            user_text
        )
        resolver_query = (
            semantic_result.routed_query
            or user_text
        )
        if semantic_result.bare_concept_routed:
            print(
                f"[semantic bare routing: {user_text!r} -> "
                f"{resolver_query!r}]"
            )
        elif semantic_result.bare_unknown_blocked:
            print(
                f"[bare unknown safety gate: "
                f"concept={semantic_result.bare_focus!r}, "
                "generation=blocked]"
            )
        knowledge_state = semantic_result.knowledge_state
        dispatch = semantic_result.dispatch
        truth_record = semantic_result.truth
        truth_result = semantic_result.truth_result

        if truth_result is not None and truth_result.warning:
            print(
                f"[truth warning: state={truth_record.state}, "
                f"concept={truth_record.concept!r}, "
                f"message={truth_result.warning}]"
            )

        if dispatch.action == "RETRIEVE":
            print(f"AI> {dispatch.answer}")
            predicate_part = (
                f", predicate_type={dispatch.predicate_type}"
                if dispatch.predicate_type
                else ""
            )
            provenance_text = (
                dispatch.provenance.compact()
                if dispatch.provenance is not None
                else "none"
            )
            truth_part = (
                f", truth_state={truth_record.state}"
                if truth_record is not None
                else ""
            )
            correction_part = (
                ", truth_correction=True"
                if truth_result is not None
                and truth_result.correction_applied
                else ""
            )
            truth_runtime_part = (
                f", truth_runtime={truth_result.runtime_status}"
                if truth_result is not None
                else ""
            )
            print(
                f"[gate=KNOWN, knowledge_state={dispatch.state}, "
                f"concept={dispatch.focus}{predicate_part}, "
                f"route={dispatch.route}, reason={dispatch.reason}, "
                f"provenance={provenance_text}"
                f"{truth_part}{correction_part}{truth_runtime_part}]"
            )
            print("[0 generated probe tokens, retrieval]")
            print()

            resolved = resolve_teaching_queue(
                Path(args.teaching_queue),
                resolver_query,
            )
            if resolved:
                print(f"[resolved teaching queue entries: {resolved}]")
            history.append((resolver_query, dispatch.answer))
            last_ai_reply = dispatch.answer
            continue

        if dispatch.action == "BLOCK":
            truth_block = (
                truth_result is not None
                and truth_result.runtime_status == "BLOCK"
            )
            stale_internalized = (
                dispatch.state == "INTERNALIZED_STALE"
                and not truth_block
            )
            if truth_block:
                route_result = "truth-state block"
                resolution_name = "TRUTH_BLOCK"
                action_name = "review/correct"
            elif stale_internalized:
                route_result = "checkpoint rebind required"
                resolution_name = "INTERNALIZED_STALE"
                action_name = "rebind/train"
            else:
                route_result = route_resolution_action(
                    args=args,
                    resolution="UNKNOWN_KNOWLEDGE",
                    action="retrieve/teach",
                    user_text=resolver_query,
                    candidate_answer="",
                    reason=(
                        f"knowledge state {dispatch.state}: "
                        f"{dispatch.reason}"
                    ),
                )
                resolution_name = "UNKNOWN_KNOWLEDGE"
                action_name = "retrieve/teach"
            block_reply = (
                truth_result.user_message
                if truth_result is not None
                and truth_result.runtime_status == "BLOCK"
                and truth_result.user_message
                else (
                    "学習証拠はありますが、現在のcheckpointには未反映です。"
                    if stale_internalized
                    else UNKNOWN_REPLY
                )
            )
            print(f"AI> {block_reply}")
            print(
                f"[knowledge-state={dispatch.state}, "
                f"concept={dispatch.focus}, generation=blocked]"
            )
            print(f"[route={route_result}]")
            if truth_block:
                print(
                    "[truth path: set a correction with "
                    "/truthset CONCEPT FALSE|OUTDATED => CORRECTION]"
                )
            elif stale_internalized:
                print(
                    "[checkpoint path: run /train to replay trusted pairs "
                    "into the current checkpoint]"
                )
            else:
                print(
                    "[teaching path: /teach ANSWER -> /train "
                    "(or /teachq QUESTION => ANSWER)]"
                )
            provenance_text = (
                dispatch.provenance.compact()
                if dispatch.provenance is not None
                else "none"
            )
            truth_part = (
                f", truth_state={truth_record.state}"
                if truth_record is not None
                else ""
            )
            truth_runtime_part = (
                f", truth_runtime={truth_result.runtime_status}"
                if truth_result is not None
                else ""
            )
            print(
                f"[gate=UNKNOWN, resolution={resolution_name}, "
                f"action={action_name}, route={dispatch.route}, "
                f"reason={dispatch.reason}, provenance={provenance_text}"
                f"{truth_part}{truth_runtime_part}]"
            )
            print("[0 generated probe tokens, 0.00s, 0.0 tok/s]")
            print()
            last_ai_reply = None
            continue

        # GENERATE: only INTERNALIZED and NON_CONCEPT reach this point.
        if dispatch.state == "INTERNALIZED":
            internalized_records = load_internalized_records(
                learning_log,
                learning_state,
            )
            internalized_record = internalized_record_for_focus(
                internalized_records,
                dispatch.focus,
            )
            print(
                "[internalized route: "
                f"concept={dispatch.focus!r}, "
                "generation=model-weights, "
                f"truth_state={truth_record.state if truth_record else 'UNVERIFIED'}]"
            )
        else:
            internalized_record = None

        user_text = resolver_query

        generation_user_text = (
            internalized_record.question
            if internalized_record is not None
            else normalize_identity_query(user_text)
        )
        prompt, selected_history = build_prompt(
            history=history,
            user_text=generation_user_text,
            history_turns=args.history_turns,
        )

        start = time.perf_counter()

        # Primary response: greedy when rejection is enabled so that the
        # displayed candidate itself is deterministic and reproducible.
        primary_temperature = 0.0 if args.unknown_rejection else args.temperature
        results = [
            generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=primary_temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                seed=0,
            )
        ]

        if args.unknown_rejection:
            for probe in range(max(0, args.probe_count - 1)):
                results.append(
                    generate_reply(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.probe_temperature,
                        top_k=args.probe_top_k,
                        repetition_penalty=args.repetition_penalty,
                        seed=1000 + probe,
                    )
                )

        primary = results[0]

        accepted = True
        confidence = primary.mean_confidence
        agreement = 1.0
        semantic_agreement = 1.0
        reason = "rejection disabled"
        qa_similarity = 0.0
        previous_similarity = -1.0
        intent = "general"
        slots: list[str] = []
        slot_coverage = 1.0

        semantic_ok = True
        semantic_reason = "semantic check disabled"

        if args.semantic_consistency:
            # Run semantic verification first. v1.5.8 may relax agreement
            # only after slots/entities/intent are semantically validated.
            (
                semantic_ok,
                qa_similarity,
                previous_similarity,
                semantic_reason,
                intent,
                slots,
                slot_coverage,
            ) = semantic_consistency_check(
                model=model,
                tokenizer=tokenizer,
                current_question=user_text,
                answer=primary.text,
                history=selected_history,
                contamination_margin=args.history_contamination_margin,
            )

        answer_concept_ok, answer_concept_reason = answer_concept_consistency(
            user_text,
            primary.text,
        )
        if semantic_ok and not answer_concept_ok:
            semantic_ok = False
            semantic_reason = answer_concept_reason

        if args.unknown_rejection:
            effective_agreement = calibrated_agreement_threshold(
                intent=intent,
                semantic_ok=semantic_ok,
                slot_coverage=slot_coverage,
                base_threshold=args.min_agreement,
            )
            (
                accepted,
                confidence,
                agreement,
                semantic_agreement,
                reason,
            ) = evaluate_unknown_gate(
                model=model,
                tokenizer=tokenizer,
                results=results,
                min_confidence=args.min_confidence,
                min_token_confidence=args.min_token_confidence,
                min_mean_margin=args.min_mean_margin,
                min_agreement=effective_agreement,
                min_semantic_agreement=args.min_semantic_agreement,
                allow_semantic_rescue=(
                    semantic_ok
                    and slot_coverage >= 1.0
                    and internalized_record is None
                ),
            )
        else:
            effective_agreement = args.min_agreement

        if accepted and args.semantic_consistency:
            if not semantic_ok:
                accepted = False
                reason = semantic_reason
            elif reason == "accepted":
                reason = semantic_reason
            # Preserve diagnostic reasons such as "semantic probe agreement".

        internalized_fidelity = -1.0
        internalized_lexical_coverage = -1.0
        internalized_required_coverage = -1.0
        internalized_contradiction = False
        internalized_missing_terms: tuple[str, ...] = ()
        if internalized_record is not None:
            composite_fidelity = composite_internalized_teacher_fidelity(
                model=model,
                tokenizer=tokenizer,
                generated_answer=primary.text,
                teacher_answer=internalized_record.teacher_answer,
                concept=internalized_record.concept,
            )
            internalized_fidelity = (
                composite_fidelity.semantic_similarity
            )
            internalized_lexical_coverage = (
                composite_fidelity.lexical_coverage
            )
            internalized_required_coverage = (
                composite_fidelity.required_coverage
            )
            internalized_contradiction = (
                composite_fidelity.contradiction
            )
            internalized_missing_terms = (
                composite_fidelity.missing_terms
            )
            fidelity_ok, fidelity_reason = (
                apply_composite_internalized_fidelity_policy(
                    accepted=accepted,
                    result=composite_fidelity,
                    semantic_threshold=args.min_internalized_fidelity,
                    lexical_threshold=(
                        args.min_internalized_lexical_coverage
                    ),
                    required_threshold=(
                        args.min_internalized_required_coverage
                    ),
                )
            )
            if not fidelity_ok:
                accepted = False
                if fidelity_reason:
                    reason = fidelity_reason
            elif accepted:
                reason = fidelity_reason

        resolution, action = classify_resolution(
            accepted,
            reason,
            user_text,
        )
        if internalized_record is not None and not accepted:
            resolution = "INTERNALIZED_UNSTABLE"
            action = "retrain/review"
            route_result = "internalized fidelity block"
        else:
            route_result = route_resolution_action(
                args=args,
                resolution=resolution,
                action=action,
                user_text=user_text,
                candidate_answer=primary.text,
                reason=reason,
            )

        if accepted and internalized_record is not None:
            route_result = "internalized model generation"

        reply = primary.text if accepted else UNKNOWN_REPLY
        new_tokens = sum(r.token_count for r in results)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        rate = new_tokens / elapsed if elapsed > 0 else 0.0

        if not reply:
            reply = UNKNOWN_REPLY if args.unknown_rejection else "(no response)"

        print(f"AI> {reply}")

        if args.show_risk and args.unknown_rejection and not accepted:
            print(f"[candidate={primary.text}]")

        if args.show_risk and args.unknown_rejection:
            if accepted and internalized_record is not None:
                status = "INTERNALIZED"
            else:
                status = "KNOWN" if accepted else "UNKNOWN"
            semantic_part = ""
            internalized_part = (
                (
                    f", internalized_concept={internalized_record.concept}"
                    f", teacher_sem={internalized_fidelity:.3f}"
                    f", teacher_lex={internalized_lexical_coverage:.3f}"
                    f", teacher_req={internalized_required_coverage:.3f}"
                    f", teacher_contra={internalized_contradiction}"
                    f", fidelity_th={args.min_internalized_fidelity:.3f}"
                    f", lex_th={args.min_internalized_lexical_coverage:.3f}"
                    f", req_th={args.min_internalized_required_coverage:.3f}"
                    f", missing={'|'.join(internalized_missing_terms) or '-'}"
                )
                if internalized_record is not None
                else ""
            )
            if args.semantic_consistency:
                slot_text = "|".join(slots) if slots else "-"
                internalized_part = (
                    (
                        f", internalized_concept={internalized_record.concept}"
                        f", teacher_fidelity={internalized_fidelity:.3f}"
                        f", fidelity_th={args.min_internalized_fidelity:.3f}"
                    )
                    if internalized_record is not None
                    else ""
                )
                semantic_part = (
                    f", intent={intent}"
                    f", slots={slot_text}"
                    f", slot_cov={slot_coverage:.2f}"
                    f", qa_sim={qa_similarity:.3f}"
                    f", prev_sim={previous_similarity:.3f}"
                    f", agr_th={effective_agreement:.2f}"
                )
            truth_part = (
                f", truth_state={truth_record.state}"
                if truth_record is not None
                else ""
            )
            print(
                f"[gate={status}, "
                f"confidence={confidence:.3f}, "
                f"min_tok_conf={primary.min_confidence:.3f}, "
                f"mean_margin={primary.mean_top2_margin:.3f}, "
                f"agreement={agreement:.3f}, "
                f"sem_agreement={semantic_agreement:.3f}"
                f"{semantic_part}"
                f"{internalized_part}"
                f"{truth_part}, "
                f"context_turns={len(selected_history)}, "
                f"resolution={resolution}, action={action}, "
                f"route={route_result}, reason={reason}]"
            )

        print(
            f"[{new_tokens} generated probe tokens, "
            f"{elapsed:.2f}s, "
            f"{rate:.1f} tok/s]"
        )
        print()

        # v1.5.4 clean-history policy:
        # only accepted KNOWN turns may enter future generation context.
        # UNKNOWN/rejected turns are intentionally discarded.
        if accepted:
            resolved = resolve_teaching_queue(
                Path(args.teaching_queue),
                user_text,
            )
            if resolved:
                print(f"[resolved teaching queue entries: {resolved}]")
            history.append((user_text, reply))
            last_ai_reply = reply
            if learning_enabled:
                print("[learning candidate ready: use /good to approve or /teach TEXT to correct]")
        else:
            last_ai_reply = None


if __name__ == "__main__":
    main()
