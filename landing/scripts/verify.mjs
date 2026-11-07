// Landing page checks. Usage: node landing/scripts/verify.mjs static|render
import { readFileSync, readdirSync, statSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

const LANDING = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(LANDING, "index.html"), "utf8");
const mode = process.argv[2];
const fail = (msg) => { console.error("FAIL:", msg); process.exit(1); };

function walk(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

if (mode === "static") {
  const banner = "Decision support only. A licensed pharmacist must review every order. Demo uses synthetic data.";
  if (!html.includes(banner)) fail("safety banner text missing or altered");
  if (!/<title>[^<]{3,60}<\/title>/.test(html)) fail("missing <title>");
  for (const id of ["demo", "how", "safety", "models", "memory", "production"]) {
    if (!html.includes(`id="${id}"`)) fail(`section #${id} missing`);
  }
  if (!html.includes("assets/demo-chart.jpg")) fail("demo image not referenced");
  if (!existsSync(join(LANDING, "assets/demo-chart.jpg"))) fail("demo image file missing");
  if (html.includes("chemo_image")) fail("page references the unredacted original photo");
  for (const m of html.matchAll(/<script[^>]*\bsrc="([^"]+)"/g)) {
    if (/^https?:/.test(m[1]) && !/@\d+\.\d+\.\d+/.test(m[1])) fail(`unpinned script ${m[1]}`);
  }
  const keyPattern = new RegExp("sk" + "-(ant|sie)-");
  for (const f of walk(LANDING)) {
    if (/\.(html|css|js|mjs|py|md|json)$/.test(f) && keyPattern.test(readFileSync(f, "utf8"))) fail(`API key-like string in ${f}`);
  }
  console.log("STATIC_OK");
} else if (mode === "render" || mode === "flow") {
  const chrome = process.env.CHROME_BIN || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
  const target = mode === "render"
    ? pathToFileURL(join(LANDING, "index.html")).href + "?run=instant"
    : pathToFileURL(join(LANDING, "scripts/flow-harness.html")).href;
  const dom = execFileSync(chrome, [
    "--headless=new", "--disable-gpu", "--no-first-run", "--allow-file-access-from-files",
    "--virtual-time-budget=5000", "--window-size=1400,1000", "--dump-dom", target,
  ], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 90000 });
  const n = (re) => (dom.match(re) || []).length;
  const checks = [];
  const expect = (name, ok, got) => checks.push([name, ok, got]);
  if (mode === "render") {
    expect("8 field rows", n(/<div class="field( sel)?( pending)?" role="option"/g) === 8, n(/<div class="field( sel)?( pending)?" role="option"/g));
    expect("6 flagged", n(/data-flagged="true"/g) === 6, n(/data-flagged="true"/g));
    const approve = dom.match(/<button[^>]*id="approve"[^>]*>/);
    expect("approve disabled", !!approve && /\bdisabled\b/.test(approve[0]), approve && approve[0]);
    expect(">=5 SIE events", n(/class="src src-sie">SIE · /g) >= 5, n(/class="src src-sie">SIE · /g));
    expect(">=3 cognee events", n(/class="src src-cog">cognee · /g) >= 3, n(/class="src src-cog">cognee · /g));
    expect(">=3 registry events", n(/class="src src-reg">Registry · /g) >= 3, n(/class="src src-reg">Registry · /g));
    expect(">=5 rules events", n(/class="src src-rules">Rules · /g) >= 5, n(/class="src src-rules">Rules · /g));
    expect("memory-verify rejected", dom.includes("REJECTED: memory proposed"), null);
    expect("9 checks settled", n(/data-check="[a-z]/g) === 9 && n(/data-state="pending"/g) === 0, `${n(/data-check="[a-z]/g)} / pending ${n(/data-state="pending"/g)}`);
    expect("guesses 0", /id="mGuess">0</.test(dom), null);
    expect(">=8 agent audit rows", n(/<span class="who">/g) >= 8, n(/<span class="who">/g));
    expect("order blocked", /id="orderChip">Blocked</.test(dom), null);
  } else {
    const m = dom.match(/<pre id="out">([^<]*)<\/pre>/);
    const r = m && m[1] !== "pending" ? JSON.parse(m[1].replace(/&quot;/g, '"')) : null;
    if (!r) fail("harness produced no result");
    console.log(JSON.stringify(r));
    expect("starts blocked", r.blockedAtStart === "Blocked" && r.approveDisabledAtStart, r.blockedAtStart);
    expect("6 resolutions", r.clicks === 6, r.clicks);
    expect("approve enabled after", r.approveEnabled, r.approveEnabled);
    expect("6 pharmacist audit rows", r.pharmacistAudit === 6, r.pharmacistAudit);
    expect("6 pii-redact calls", r.piiRedact === 6, r.piiRedact);
    expect("6 cognee writes", r.cogneeWrites === 6, r.cogneeWrites);
    expect("2 memory feedback events", r.feedback === 2, r.feedback);
    expect("ready to approve", r.readyChip === "Ready to approve", r.readyChip);
    expect("approved", r.approvedChip === "Approved" && r.fhir >= 1, r.approvedChip);
    expect("reset clears", r.afterResetDisabled && r.afterResetFeedEmpty, null);
    expect("re-run blocks again", r.rerunChip === "Blocked" && r.rerunFlagged === 6, r.rerunChip);
  }
  let ok = true;
  for (const [name, pass, got] of checks) { console.log(`${pass ? "ok  " : "FAIL"} ${name}${got !== null && got !== undefined ? ` (${got})` : ""}`); ok &&= pass; }
  if (!ok) fail(`${mode} checks failed`);
  console.log(mode === "render" ? "RENDER_OK" : "FLOW_OK");
} else if (mode === "live") {
  // Start the backend in mock mode (replays a recorded live SIE run; no credits), load the page from it,
  // let it upload the sample over HTTP and stream the run over SSE, then inspect the DOM.
  const { spawn } = await import("node:child_process");
  const { mkdtempSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const ROOT = join(LANDING, "..");
  const port = 8700 + Math.floor(Math.random() * 200);
  const srv = spawn(join(ROOT, ".venv/bin/python"), ["-m", "uvicorn", "app.main:app", "--app-dir", join(ROOT, "backend"), "--port", String(port), "--log-level", "warning"], {
    env: { ...process.env, MODEL_BACKEND: "mock", MEMORY_BACKEND: "local", PILLSPARK_REPLAY_SPEED: "0.02",
           PILLSPARK_DATA_DIR: mkdtempSync(join(tmpdir(), "pillspark-ui-")), APP_ORIGIN: `http://127.0.0.1:${port}` }, stdio: "ignore" });
  try {
    let up = false;
    for (let i = 0; i < 60 && !up; i++) {
      try { up = (await fetch(`http://127.0.0.1:${port}/healthz`)).ok; } catch { await new Promise((r) => setTimeout(r, 250)); }
    }
    if (!up) fail("backend did not start");
    const chrome = process.env.CHROME_BIN || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    const dom = execFileSync(chrome, ["--headless=new", "--disable-gpu", "--no-first-run", "--virtual-time-budget=15000", "--window-size=1400,1000",
      "--dump-dom", `http://127.0.0.1:${port}/?autorun=sample`], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 120000 });
    const n = (re) => (dom.match(re) || []).length;
    const checks = [
      ["live badge", /id="modeBadge"[^>]*>(Live · Superlinked SIE|Replay of a recorded SIE run)</.test(dom)],
      ["feed has SIE LightOnOCR event", n(/class="src src-sie">SIE · LightOnOCR</g) >= 1],
      ["feed has SIE Qwen event", n(/class="src src-sie">SIE · Qwen</g) >= 1],
      ["feed has registry event", n(/class="src src-reg">Registry · RxNorm</g) >= 1],
      ["feed has memory event", n(/class="src src-(mem|cog)">/g) >= 1 + 1],
      [">10 server field rows", n(/<div class="field( sel)?" role="option"/g) > 10],
      ["flagged fields present", n(/data-flagged="true"/g) > 0],
      ["order chip Blocked", /id="orderChip">Blocked</.test(dom)],
      ["six stages done", n(/<li data-st="\d"[^>]*class="done"/g) === 6],
      ["audit rows from server", n(/<span class="who">/g) >= 2],
      ["checks settled", n(/data-check="[a-z]/g) === 9 && n(/data-state="pending"/g) === 0],
      ["field boxes drawn on the chart", n(/<div class="box [^"]*"/g) >= 1],
    ];
    let ok = true;
    for (const [name, pass] of checks) { console.log(`${pass ? "ok  " : "FAIL"} ${name}`); ok &&= pass; }
    if (!ok) fail("live UI checks failed");
    console.log("LIVE_UI_OK");
  } finally { srv.kill(); }
} else {
  fail("mode must be static or render");
}
