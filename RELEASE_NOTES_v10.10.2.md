# LLM_TRY v10.10.2 — Provenance / Source Tracking

v10.10.2 adds auditable provenance metadata to every resolved knowledge state.

## Provenance model

~~~text
source
origin
timestamp
fingerprint
retrieval_priority
evidence
metadata
~~~

Missing historical metadata is left empty rather than inferred.

## Priority

~~~text
TYPED           10
CANONICAL       20
UNIFIED         30
INTERNALIZED    40
RAW_CORPUS_ONLY 50
UNKNOWN         60
NON_CONCEPT     70
~~~

The priority is descriptive of the Resolver order. It does not modify gate
thresholds.

## Examples

~~~text
TYPED
  source = typed-subject-proposition
  origin = data/typed_subject_propositions_v10100.jsonl

CANONICAL
  source = canonical-definition
  origin = chat.py:CANONICAL_DEFINITIONS

UNIFIED
  source = atomic-proposition or stored row source
  origin = data/unified_semantic_memory_v1090.jsonl
  timestamp = updated_at when available

INTERNALIZED
  source = chat-manual / chat-approved / chat-recovery
  origin = data/chat_history.jsonl
  timestamp = teaching timestamp when available
  fingerprint = trained pair fingerprint

RAW_CORPUS_ONLY
  source = raw-corpus
  origin = data/data-nagato.txt
~~~

## Runtime inspection

~~~text
/provenance GPUの性質は
/provenance CPUとは
/provenance 宇宙とは
/provenance 量子センサーとは
/provenance 時間とは
~~~

The /dispatch command also includes compact provenance in its diagnostic line.

## Unified semantic timestamps

New or updated Unified Semantic Memory rows now receive an updated_at timestamp.
Existing rows without a timestamp remain valid and are reported with an empty
timestamp until rewritten.

## Verification

~~~powershell
python .\run_knowledge_provenance_v10102.py
python .\run_knowledge_provenance_end_to_end_v10102.py
~~~

Expected statuses:

~~~text
KNOWLEDGE_PROVENANCE_PASS
KNOWLEDGE_PROVENANCE_END_TO_END_PASS
~~~
