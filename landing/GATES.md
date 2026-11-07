# Gates: PillSpark landing page

OWNS: landing/**

Scope: A self-contained landing page (landing/index.html + assets) that pitches PillSpark and plays a labelled, real-time replay of the six-agent pipeline on the real handwritten chemo chart (patient identifiers redacted), showing Superlinked SIE calls, cognee memory reads/writes, safety checks, monitoring, flagging and an append-only audit trail.

- [x] G1: index.html carries the exact safety banner, every planned section, the redacted image reference, no API keys, and no unpinned external scripts
  CHECK: node landing/scripts/verify.mjs static
  EXPECT: STATIC_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=8fe620905b5feb82857d8e4c32f45b919eeb4487f3771e4fee5e45f364441bde; exit=0; EXPECT=matched; output-sha256=294266116f414ab83f14296070a0cf4af3d89a874e8aa214c672de29117b6efd; output-bytes=10; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G2: The published demo image has patient name and reg. no. redacted in pixels (flat fill), while the same check rejects the unredacted original
  CHECK: python3 landing/scripts/check_redaction.py
  EXPECT: REDACTION_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=24f0f875356d54cf089b3d1533fda6755dd6feb4f0563976619d880122f97d1e; exit=0; EXPECT=matched; output-sha256=223d9dcd8aa3001b6a358a8ea22fae43131e1f34fdf58ea0f26272202c771714; output-bytes=145; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G3: After a completed run (?run=instant) the page shows 8 fields with 6 flagged, Approve disabled, an activity feed with SIE, cognee, registry and rules events, the memory-verify rejection, 9 settled safety checks, guesses 0, and agent-written audit rows
  CHECK: node landing/scripts/verify.mjs render
  EXPECT: RENDER_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=9632fab9e587f5d72200eb6c72a20300a34f66916c09fd8c8ce845e0f2b1efd3; exit=0; EXPECT=matched; output-sha256=af201e87e524d45a33d0e54de8420f999f7bd77fbb166d294018ecd9b5653aa8; output-bytes=391; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G4: Resolving every flag as a pharmacist enables Approve, writes pharmacist audit rows, a PII-redaction SIE call and a cognee write per resolution, cognee feedback for the memory-suggested field, and Reset/Run returns to a blocked order
  CHECK: node landing/scripts/verify.mjs flow
  EXPECT: FLOW_OK
  EVIDENCE: automatic-evidence=v1; definition-sha256=8de0bec1d36c92c581c53182806d277f047e95e2c58a63641d03cf6abb5e88ef; exit=0; EXPECT=matched; output-sha256=8207b591026248debc11085ae318545f5c380ae6f9403cf56fe18d2cfeefe182; output-bytes=669; shell=/bin/sh; cwd=/Users/rodela/Desktop/Desktop/GitProjects/Hackathon_superlinked/pillspark; path=850bd16ad3bb/21 entries

- [x] G5: Visual review of desktop (1440px) mid-run and finished, and phone (390px): calm clinical look, sponsor badges legible, no overlap, no horizontal scroll, boxes land on the right handwriting
  EVIDENCE: 2026-11-07 reviewed headless-Chrome screenshots: 1440px at 7.4s mid-run (Verification active, 15 events, BSA dashed box on the blank BSA line, checks half settled) and finished (?run=instant: 26 events, REJECTED memory-verify row, 4 checks need review, invariant enforced, latency bars); 390px via iframe (panels stack, feed rows wrap to two lines, badges legible, no clipping). Found and fixed: cognee label doubled, empty space under feed, replay loop on requestAnimationFrame stalling in headless (moved to setTimeout; DOM diagnostic at 3s/7.5s/12s = 6/16/26 events, OCR flash cleared by 7.5s). Stage highlight pass: harness clicks Process and samples at 0.3/2/5.5/8/10/12.5s -> active = Intake/Extraction/Consensus/Verification/Triage/none, page scroll delta 0 at every sample (field-list scrolling no longer moves the page); screenshots show active stage tinted in its owner colour (SIE indigo, cognee magenta, teal otherwise) with live timer, done stages green with measured durations, pending stages dimmed.
