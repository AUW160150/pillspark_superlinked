# Gates: PillSpark real-time document intake

OWNS: backend/**, landing/**, GATES.md

Scope: A running FastAPI service that accepts an uploaded prescription or chart image, queues it, runs the six agents with real Superlinked SIE models (two OCR readers, GLiNER2, Qwen vision structured parse, Qwen3 reranker, pii-redact) plus RxNorm/NPPES and deterministic checks, streams every event live to the landing page over server-sent events, persists orders and an append-only audit trail, lets a pharmacist resolve flags and approve, and records corrections in memory (cognee when available) without patient identifiers, while never letting a model or memory verify a field.

- [x] G1: Unit tests pass (pytest exits 0): set_status() rejects "verified" from model/memory/reranker sources and only a pharmacist resolves, unreadable withholds the value, PHI is stripped structurally and by the PII model before memory writes, dataset write permissions, recall and retraction, NPI/DEA checksums, BSA dose pause, ISMP q.d., FHIR R4 MedicationRequest shape
  CHECK: .venv/bin/python -m pytest -q backend/tests
  EXPECT: passed in
  EVIDENCE: automatic-evidence=v1; definition-sha256=439d3a6103690da7565901492de9e0bca69d7b4f93cf3da6ebb9c15b33818900; exit=0; EXPECT=matched; output-sha256=bbb720f7f870fe62bbb8a7d0478e52d8253d4fc0386bc15553d530b34bc19537; output-bytes=99; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G2: Mock-mode end to end over HTTP: upload a sample, SSE stream delivers all six stages and a final order event, order is blocked with flags, resolving every flag then approving yields approved with pharmacist audit rows and memory writes, and a duplicate upload with the same idempotency key returns the same order
  CHECK: .venv/bin/python backend/scripts/e2e.py --mode mock
  EXPECT: E2E_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=c36017f0a02ea320395cedee5b14ac6cbdefdcf002549113575eea18332eccd4; exit=0; EXPECT=matched; output-sha256=91fec4460364736801011436a6c40f1eccffd8f35a4d2b149ab082bd495bc38c; output-bytes=8088; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G3: Live SIE run on the redacted chemo chart: real calls to LightOnOCR, GLM-OCR, GLiNER2, Qwen vision and Qwen3-Reranker stream as events with latency and credits, the order ends blocked, and no field whose readers disagree is verified (guesses = 0)
  CHECK: .venv/bin/python backend/scripts/e2e.py --mode live
  EXPECT: LIVE_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=91ba95eac9bed6a0a159137ed33f40a605e92f126afdf15c453d727c3cae6822; exit=0; EXPECT=matched; output-sha256=ad9e56bde466849c67ff139b6db6cb00cadc6bf0ce152935125f443303c03b2c; output-bytes=7959; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G4: The landing page served by the backend runs a real intake from the browser: clicking "Use sample chart" uploads, the feed fills from server events, stages advance, fields render from the server, and the order chip ends Blocked
  CHECK: node landing/scripts/verify.mjs live
  EXPECT: LIVE_UI_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=1e3954613203c66c06f78748595bb07a2f17d4b9566cb89d44fa3fd58422a8b1; exit=0; EXPECT=matched; output-sha256=289f130e45b22c4599a64a204c6ae8faa9c1d9dd2d07ab1c120203c1b5457482; output-bytes=331; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G5: Landing static/replay checks still pass (banner, redaction, replay render and pharmacist flow)
  CHECK: node landing/scripts/verify.mjs static && node landing/scripts/verify.mjs render && node landing/scripts/verify.mjs flow
  EXPECT: FLOW_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=f1e4332e017ce433adec3da608f40d9cfa130d2097c7dbf137a24c7cb61eba76; exit=0; EXPECT=matched; output-sha256=3ea91613c15ed086085159a0bdea22cf818add70c051d8a37ed102102f26ba0a; output-bytes=1070; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G7: Learning loop: after a pharmacist resolves an order, a second intake of the same document shows "Seen before" memory hints, every memory attempt to verify is rejected by set_status(), nothing is verified by a non-rules source, and the order stays blocked
  CHECK: .venv/bin/python backend/scripts/learn_check.py
  EXPECT: LEARN_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=22c48984fa9ff8785e1a2ae67d9db9ad9ec4377d590aec18de897f8911c87103; exit=0; EXPECT=matched; output-sha256=29954966d1eee3668c8261db98c663496989a803c7cd539fcbde123d059c314b; output-bytes=678; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G8: cognee multi-user memory on SIE: five users and three permissioned datasets, a pharmacist correction stored in cognee, a technician write to pharmacist_corrections refused by cognee itself, background cognify completes with an SIE model
  EVIDENCE: 2026-11-07 smoke run (throwaway data dir, MEMORY_BACKEND=cognee): init 4.3s, users [extraction-agent, verification-agent, triage-agent, pharmacist, technician], datasets pharmacist_corrections/prescriber_patterns/agent_notes with ids; pharmacist write stored_in=cognee; technician add to pharmacist_corrections -> cognee PermissionDeniedError (403); cognify_runs=1, last_error=null with LLM openai/Qwen/Qwen3.8-27B-FP8 + embeddings BAAI/bge-m3 on SIE (GLM-5.3-Flash rejected: no json_schema grammar on SIE). Full mock e2e with MEMORY_BACKEND=cognee: E2E_OK, 26 corrections written, no request stall after moving writes off the event loop. Recall still matches the local index (labelled "memory index" in events); writes are labelled cognee only when cognee stored them.

- [x] G6: Visual review of a live run in the browser (desktop): live badge, real model names/latency/credits in the feed, fields grouped by order and medication lines with OCR evidence snippets, flags and audit readable
  EVIDENCE: 2026-11-07 headless-Chrome screenshots of the page served by the backend (?autorun=sample, backend in mock mode so SIE outputs are a recorded live run; same UI path as live, events tagged REPLAYED): mid-run (Extraction stage tinted SIE indigo, feed streaming LightOnOCR 7.41s / GLM-OCR 2.58s) and finished (order Blocked, ORDER group header, LightOnOCR evidence snippet on Patient name, candidate chips + "Enter value… reason required", checks 5 need review + invariant, monitoring 14 SIE calls · 1147 credits, per-model latency bars, audit rows). Fixed during review: Skip button visible in live mode ([hidden] overridden by .btn), stale "Waiting for a document" text, memory tile label claiming cognee when local, audit trail missing field decisions, 59s cold verification (parallel RxNorm prefetch -> mock e2e 8.0s -> 3.2s).
