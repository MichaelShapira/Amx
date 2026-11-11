import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, defaultWizard, fmt, type Candidate, type Plan, type Proposal, type Side, type UseCase, type Wizard } from "../api";
import { useCatalog } from "../App";
import DataPreview from "../components/DataPreview";
import { Badge, Chips, Help, Label, Note, Reasons, Spinner, Tile, Toggle, instanceLine } from "../components/ui";

const STEPS = ["Use case", "Test data", "Depth", "Models", "Hardware", "Load & targets", "Review"];

export default function NewRun() {
  const cat = useCatalog();
  const nav = useNavigate();
  const [step, setStep] = useState(0);
  const [w, setW] = useState<Wizard>(() => {
    try { const s = sessionStorage.getItem("wizard"); return s ? { ...defaultWizard, ...JSON.parse(s) } : defaultWizard; } catch { return defaultWizard; }
  });
  useEffect(() => { sessionStorage.setItem("wizard", JSON.stringify(w)); }, [w]);
  const set = (patch: Partial<Wizard>) => setW((p) => ({ ...p, ...patch }));

  if (!cat) return <Spinner label="Loading the catalog…" />;

  const canNext = [
    true,
    true,
    true,
    w.models.length > 0,
    !!w.gpu || w.cpus.length > 0,
    w.request_mode === "single" || w.request_mode === "batch" ? true : w.concurrency.length > 0,
    true,
  ][step];

  return (
    <div>
      <div className="topbar">
        <div><h1>New benchmark</h1><div className="muted">Seven steps; nothing starts until you press Start on the last one.</div></div>
        <button className="btn sm" onClick={() => { setW(defaultWizard); setStep(0); }}>Reset</button>
      </div>
      <div className="steps">
        {STEPS.map((s, i) => (
          <div key={s} className={`step ${i === step ? "current" : ""} ${i < step ? "done" : ""}`} onClick={() => i < step && setStep(i)}>
            <span className="n">{i < step ? "✓" : i + 1}</span>{s}
          </div>
        ))}
      </div>
      <div className="card">
        {step === 0 && <StepUseCase w={w} set={set} />}
        {step === 1 && <StepData w={w} set={set} />}
        {step === 2 && <StepDepth w={w} set={set} />}
        {step === 3 && <StepModels w={w} set={set} />}
        {step === 4 && <StepHardware w={w} set={set} />}
        {step === 5 && <StepLoad w={w} set={set} />}
        {step === 6 && <StepReview w={w} set={set} onStarted={(id) => nav(`/runs/${id}`)} />}
        <div className="wizard-nav">
          <button className="btn" disabled={step === 0} onClick={() => setStep(step - 1)}>← Back</button>
          {step < STEPS.length - 1 && <button className="btn primary" disabled={!canNext} onClick={() => setStep(step + 1)}>Next →</button>}
        </div>
      </div>
    </div>
  );
}

type SP = { w: Wizard; set: (p: Partial<Wizard>) => void };

function StepUseCase({ w, set }: SP) {
  const cat = useCatalog()!;
  const pick = (uc: UseCase) => {
    const defaults = uc === "dspm" ? { concurrency: [1, 8, 16], sla: {} } : { concurrency: [1, 4, 16], sla: { p99_ttft_ms: 2000, p50_tpot_ms: 150 } };
    set({ use_case: uc, models: [], ...defaults });
  };
  return (
    <div>
      <h2>1 · What do you want to size hardware for?</h2>
      <p className="muted">The two use cases use different test data and different success criteria; a run covers one.</p>
      <div className="tiles">
        {(Object.entries(cat.use_cases) as [UseCase, any][]).map(([k, u]) => (
          <Tile key={k} selected={w.use_case === k} onClick={() => pick(k)} title={u.name} desc={<><div>{u.short}</div><div className="tiny" style={{ marginTop: 6 }}>{u.long}</div></>} />
        ))}
      </div>
    </div>
  );
}

function StepData({ w, set }: SP) {
  const cat = useCatalog()!;
  return (
    <div>
      <h2>2 · Which test data?</h2>
      {w.use_case === "standard" ? (
        <>
          <p className="muted">Fixed prompts from public-domain books, cut to a length and wrapped in each model's chat format. Pick the input length; every request generates the same number of answer tokens so timings are comparable.</p>
          <div className="tiles mb">
            {Object.entries(cat.prompt_sets).map(([k, p]) => <Tile key={k} selected={w.prompt_set === k} onClick={() => set({ prompt_set: k })} title={p.label} desc={p.help} />)}
          </div>
          <Label help="How many tokens (roughly words) each answer generates. Every request generates exactly this many so results are comparable; 128 is a short paragraph.">Answer length (tokens)</Label>
          <Chips options={[64, 128, 256, 512]} value={[w.answer_tokens]} onChange={(v) => set({ answer_tokens: v.find((x) => x !== w.answer_tokens) ?? w.answer_tokens })} />
        </>
      ) : (
        <>
          <p className="muted">A synthetic dataset of 3,000 documents with known answers. Pick the detection scenarios to run; each uses its own slice of the dataset.</p>
          <div className="tiles mb">
            {Object.entries(cat.workloads).map(([k, p]) => (
              <Tile key={k} selected={w.workloads.includes(k)} onClick={() => set({ workloads: w.workloads.includes(k) ? w.workloads.filter((x) => x !== k) : [...w.workloads, k].sort() })} title={p.label} desc={p.help} />
            ))}
          </div>
          <Label help="How many of the 3,000 test documents each option classifies for the accuracy figures. 300 gives a quick read with wide confidence intervals; 3,000 gives decision-grade figures.">Documents in the accuracy check</Label>
          <Chips options={[300, 1000, 3000]} value={[w.test_items]} onChange={(v) => set({ test_items: v.find((x) => x !== w.test_items) ?? w.test_items })} />
        </>
      )}
      <div className="hr" />
      <h3>Preview <span className="muted small">— real examples of what the models will see</span></h3>
      <DataPreview useCase={w.use_case} promptSet={w.use_case === "standard" ? w.prompt_set : undefined} workloads={w.use_case === "dspm" ? w.workloads : undefined} compact model={w.models[0]} />
    </div>
  );
}

function StepDepth({ w, set }: SP) {
  const cat = useCatalog()!;
  return (
    <div>
      <h2>3 · Quick check or full test?</h2>
      <p className="muted">More requests give steadier percentiles; three passes show whether a result repeats.</p>
      <div className="tiles mb">
        {Object.entries(cat.depths).map(([k, d]) => (
          <Tile key={k} selected={w.depth === k} onClick={() => set({ depth: k as any })} title={d.label} desc={d.help} badge={<Badge kind={k === "full" ? "ok" : "warn"}>{k === "full" ? "decision-grade" : "SMOKE label"}</Badge>} />
        ))}
        <Tile selected={w.depth === "custom"} onClick={() => set({ depth: "custom", requests: w.requests ?? 200, passes: w.passes ?? 3 })} title="Custom" desc="Set the request count and passes yourself.">
          {w.depth === "custom" && (
            <div className="row mt" onClick={(e) => e.stopPropagation()}>
              <label className="small">Requests per level <input type="number" min={1} max={100000} value={w.requests ?? 200} onChange={(e) => set({ requests: Number(e.target.value) })} style={{ width: 90 }} /></label>
              <label className="small">Passes <input type="number" min={1} max={10} value={w.passes ?? 3} onChange={(e) => set({ passes: Number(e.target.value) })} style={{ width: 70 }} /></label>
            </div>
          )}
        </Tile>
      </div>
      <Note>{w.depth === "full" || (w.depth === "custom" && (w.requests ?? 0) >= 200 && (w.passes ?? 0) >= 3) ? "This run will be decision-grade." : "This run will be a quick check: every number in the report carries the SMOKE label and must not drive a buying decision."}</Note>
    </div>
  );
}

function StepModels({ w, set }: SP) {
  const cat = useCatalog()!;
  const models = cat.models.filter((m) => m.use_cases.includes(w.use_case));
  const toggle = (id: string) => set({ models: w.models.includes(id) ? w.models.filter((x) => x !== id) : [...w.models, id] });
  return (
    <div>
      <h2>4 · Which models?</h2>
      <p className="muted">Each model runs on every hardware option you pick next, so the comparison is like for like. Size is the main driver of speed; the license says whether a commercial decision may rest on the result.</p>
      <div className="tiles">
        {models.map((m) => (
          <Tile key={m.id} selected={w.models.includes(m.id)} onClick={() => toggle(m.id)} title={m.name}
            badge={<><Badge>{m.params_b}B</Badge><Badge kind={m.commercial_ok ? "ok" : "warn"}>{m.commercial_ok ? m.license : "research license"}</Badge></>}
            desc={<><div><b>{m.kind}</b>{m.batch_capable ? " · supports batched requests" : ""}</div><div>{m.description}</div></>} />
        ))}
      </div>
      {w.models.some((id) => !cat.models.find((m) => m.id === id)?.commercial_ok) && (
        <Note kind="warn" ><span>One of the chosen models has a research-only license: its results are for evaluation, not for a commercial decision on that model.</span></Note>
      )}
    </div>
  );
}

function StepHardware({ w, set }: SP) {
  const cat = useCatalog()!;
  const gpuHw = cat.hardware.filter((h) => h.accelerator === "gpu");
  const cpuHw = cat.hardware.filter((h) => h.accelerator === "cpu");
  const [prop, setProp] = useState<Proposal | null>(null);
  const [busy, setBusy] = useState(false);
  const chosenModels = cat.models.filter((m) => w.models.includes(m.id));

  const precsFor = (hwId: string) => {
    const hw = cat.hardware.find((h) => h.id === hwId)!;
    const set_ = new Set<string>();
    for (const m of chosenModels) for (const [p, spec] of Object.entries(m.precisions)) if (spec.engines.some((e) => hw.engines.includes(e)) && !spec.quality_only) set_.add(p);
    return [...set_];
  };
  const toggleGpu = (hw: typeof gpuHw[number]) => set({ gpu: w.gpu?.hardware === hw.id ? null : { hardware: hw.id, instance_type: w.gpu?.instance_type ?? hw.default_instance_type, precisions: ["bf16"], alternates: hw.alternates } });
  const toggleCpu = (hw: typeof cpuHw[number]) => {
    const on = w.cpus.find((c) => c.hardware === hw.id);
    set({ cpus: on ? w.cpus.filter((c) => c.hardware !== hw.id) : [...w.cpus, { hardware: hw.id, instance_type: w.cpus[0]?.instance_type ?? hw.default_instance_type, precisions: ["bf16"], alternates: hw.alternates }] });
  };
  const setSide = (hwId: string, patch: Partial<Side>) => {
    if (w.gpu?.hardware === hwId) set({ gpu: { ...w.gpu, ...patch } });
    else set({ cpus: w.cpus.map((c) => (c.hardware === hwId ? { ...c, ...patch } : c)) });
  };
  const propose = async () => {
    setBusy(true);
    try { setProp(await api.propose({ models: w.models, precision: w.precision, gpu_type: w.gpu?.instance_type ?? null, cpu_type: w.cpus[0]?.instance_type ?? null })); }
    finally { setBusy(false); }
  };
  const gpuTypes = Object.values(cat.instances).filter((s) => s.accelerator === "gpu").sort((a, b) => a.usd_per_hour - b.usd_per_hour);
  const cpuTypes = Object.values(cat.instances).filter((s) => s.accelerator === "cpu").sort((a, b) => a.usd_per_hour - b.usd_per_hour);
  const pair = prop?.pairing ?? null;

  const sideBlock = (hw: typeof cat.hardware[number], side: Side | undefined, kind: "gpu" | "cpu") => (
    <div key={hw.id} className={`tile ${kind} ${side ? "selected" : ""}`} style={{ cursor: "default" }}>
      <span className="check" onClick={() => (kind === "gpu" ? toggleGpu(hw as any) : toggleCpu(hw as any))} style={{ cursor: "pointer" }}>{side ? "✓" : ""}</span>
      <div className="t" onClick={() => (kind === "gpu" ? toggleGpu(hw as any) : toggleCpu(hw as any))} style={{ cursor: "pointer" }}>
        <Badge kind={kind}>{kind.toUpperCase()}</Badge> {kind === "gpu" ? "NVIDIA" : "Intel Xeon with AMX"} · {hw.software.join(" + ")}
        <Help text={hw.software_help} />
      </div>
      {side && (
        <div className="mt">
          <Label help="The AWS instance size for this option. The report prices results with the On-Demand rate of the size that ran. Press Propose below for suggestions with reasons.">Instance size</Label>
          <select value={side.instance_type} onChange={(e) => setSide(hw.id, { instance_type: e.target.value })} style={{ width: "100%" }}>
            {(kind === "gpu" ? gpuTypes : cpuTypes).map((s) => <option key={s.type} value={s.type}>{s.type} · {instanceLine(s, true)}</option>)}
          </select>
          <Label help="Number formats to test on this hardware. BF16 is the like-for-like baseline; compressed formats (INT8, INT4) are faster and cheaper on CPUs and the report shows what they cost in answer quality.">Number formats</Label>
          <Chips options={precsFor(hw.id)} value={side.precisions} onChange={(v) => setSide(hw.id, { precisions: v.length ? v : ["bf16"] })} labels={(p) => cat.precisions[p]?.label ?? p} />
        </div>
      )}
    </div>
  );

  return (
    <div>
      <h2>5 · Which hardware to compare?</h2>
      <p className="muted">Tick a GPU option and one or more CPU options (different serving software is a different option). Each becomes one test instance that runs every model you picked. Then pick an instance size, or let the tool propose sizes and explain why.</p>
      <div className="grid c2">
        <div>
          <h3><Badge kind="gpu">GPU side</Badge></h3>
          {gpuHw.map((hw) => sideBlock(hw, w.gpu?.hardware === hw.id ? w.gpu : undefined, "gpu"))}
        </div>
        <div>
          <h3><Badge kind="cpu">CPU side</Badge></h3>
          <div className="grid" style={{ gap: 10 }}>{cpuHw.map((hw) => sideBlock(hw, w.cpus.find((c) => c.hardware === hw.id), "cpu"))}</div>
        </div>
      </div>
      <div className="hr" />
      <div className="row between">
        <div>
          <h3 style={{ margin: 0 }}>Propose instance sizes <Help text="Suggestions for each side, judged against your models and the other side's current choice: does the model fit in memory with room for the KV cache, is the hourly price within the fair band (half to double) of the other side, is it a single GPU, how many cores feed AMX, is it the current generation, and has this size had capacity problems in this zone recently." /></h3>
          <div className="small muted">Based on {chosenModels.map((m) => m.name).join(", ") || "the chosen models"}{w.cpus[0] ? ` and the CPU choice ${w.cpus[0].instance_type}` : ""}{w.gpu ? ` and the GPU choice ${w.gpu.instance_type}` : ""}.</div>
        </div>
        <button className="btn primary" onClick={propose} disabled={busy || w.models.length === 0}>{busy ? <Spinner label="Thinking…" /> : "Propose"}</button>
      </div>
      {pair && (
        <Note kind={pair.fair ? "ok" : "warn"}><b>{pair.fair ? "Fair comparison" : "Not a fair comparison"}</b> — {pair.reason}.</Note>
      )}
      {prop && (
        <div className="grid c2 mt">
          <CandidateList title="GPU sizes" kind="gpu" list={prop.gpu} current={w.gpu?.instance_type} onPick={(t) => { if (w.gpu) setSide(w.gpu.hardware, { instance_type: t }); else set({ gpu: { hardware: gpuHw[0].id, instance_type: t, precisions: ["bf16"], alternates: gpuHw[0].alternates } }); }} />
          <CandidateList title="CPU sizes" kind="cpu" list={prop.cpu} current={w.cpus[0]?.instance_type} onPick={(t) => { if (w.cpus.length) set({ cpus: w.cpus.map((c) => ({ ...c, instance_type: t })) }); else set({ cpus: [{ hardware: cpuHw[0].id, instance_type: t, precisions: ["bf16"], alternates: cpuHw[0].alternates }] }); }} />
        </div>
      )}
      <div className="hr" />
      <Label help="Run every hardware option at the same time (fastest; one instance per option is up at once) or one after another (fewer instances running at any moment; the run takes longer). Results are the same either way.">How to run the hardware options</Label>
      <Toggle value={w.parallel_targets ? "par" : "seq"} onChange={(v) => set({ parallel_targets: v === "par" })} options={[{ value: "par", label: "All in parallel" }, { value: "seq", label: "One after another" }]} />
      <div className="mt">
        <Label help="Spot: spare capacity at a discount that AWS may reclaim with two minutes' notice; the run then launches a replacement and re-runs only the unfinished measurements. On-Demand: guaranteed for the run, full price. Cost columns always use On-Demand prices so results stay comparable.">How to buy the test instances</Label>
        <Toggle value={w.purchase} onChange={(v) => set({ purchase: v })} options={[{ value: "spot", label: "Spot (cheaper, may be interrupted)" }, { value: "on-demand", label: "On-Demand (guaranteed)" }]} />
        {w.purchase === "spot" && <label className="row small mt"><input type="checkbox" checked={w.spot_fallback} onChange={(e) => set({ spot_fallback: e.target.checked })} /> Fall back to On-Demand if Spot has no capacity for 10 minutes</label>}
      </div>
    </div>
  );
}

function CandidateList({ title, kind, list, current, onPick }: { title: string; kind: "gpu" | "cpu"; list: Candidate[]; current?: string; onPick: (t: string) => void }) {
  const fitBadge = (f: Candidate["fit"]) => f === "recommended" ? <Badge kind="ok">recommended</Badge> : f === "possible" ? <Badge kind="warn">possible</Badge> : <Badge kind="bad">not advisable</Badge>;
  return (
    <div>
      <h3><Badge kind={kind}>{title}</Badge></h3>
      {list.map((c) => (
        <div key={c.instance_type} className={`tile ${kind} ${current === c.instance_type ? "selected" : ""}`} style={{ marginBottom: 10, cursor: "default" }}>
          <div className="row between">
            <div className="t">{c.instance_type} {fitBadge(c.fit)}</div>
            <button className="btn sm" onClick={() => onPick(c.instance_type)} disabled={current === c.instance_type}>{current === c.instance_type ? "Selected" : "Use this"}</button>
          </div>
          <div className="d">{instanceLine(c.spec)}</div>
          <div className="mt"><Reasons reasons={c.reasons} /></div>
        </div>
      ))}
    </div>
  );
}

function StepLoad({ w, set }: SP) {
  const cat = useCatalog()!;
  const chosen = cat.models.filter((m) => w.models.includes(m.id));
  const batchOk = chosen.length > 0 && chosen.every((m) => m.batch_capable);
  const levels = [1, 2, 4, 8, 10, 16, 20, 32, 50, 64];
  return (
    <div>
      <h2>6 · How hard to push, and what counts as acceptable?</h2>
      <Label help="A single request shows each machine's best-case latency. Concurrent requests keep N in flight, as a service under load. Batched requests put several inputs in one request; available for classifier and detector models (chat models take one prompt per request and the server batches concurrent requests itself).">Request shape</Label>
      <div className="tiles mb">
        {Object.entries(cat.request_modes).map(([k, m]) => (
          <Tile key={k} selected={w.request_mode === k} disabled={k === "batch" && !batchOk} onClick={() => set({ request_mode: k as any, concurrency: k === "single" ? [1] : w.concurrency.length ? w.concurrency : [1, 4, 16] })} title={m.label} desc={m.help + (k === "batch" && !batchOk ? " Not available for the chosen chat models." : "")} />
        ))}
      </div>
      {w.request_mode !== "single" && (
        <>
          <Label help="How many requests are kept in flight at the same time, per hardware option. Each level is a separate measurement. Latency rises and throughput rises with concurrency; the verdict looks for the level where the machine still meets your targets.">Concurrent requests (pick the levels to measure)</Label>
          <Chips options={levels} value={w.concurrency} onChange={(v) => set({ concurrency: v.length ? v : [1] })} labels={(c) => cat.concurrency_hints[String(c)] ? `${c} · ${cat.concurrency_hints[String(c)]}` : String(c)} />
        </>
      )}
      {w.request_mode === "batch" && (
        <div className="mt">
          <Label help="Inputs per request. Classifiers score a batch in one pass; larger batches raise throughput and latency together.">Inputs per request (batch sizes)</Label>
          <Chips options={[1, 4, 8, 16, 32, 64]} value={w.batch_sizes} onChange={(v) => set({ batch_sizes: v.length ? v : [8] })} />
        </div>
      )}
      <div className="hr" />
      <h3>What counts as acceptable? <Help text="The service levels a hardware option must meet to count. The report marks every measurement that meets them, then picks the fastest GPU and the fastest CPU among those. Capacity (can the CPU side match the GPU's throughput with a reasonable number of machines) decides the verdict; cost is shown alongside." /></h3>
      {w.use_case === "standard" ? (
        <div className="grid c3">
          <div><Label help="The longest acceptable wait before the answer starts appearing, for 99 of 100 requests. 2,000 ms is a common chat target. Empty = no limit.">First word within (ms)</Label>
            <input type="number" value={w.sla.p99_ttft_ms ?? ""} onChange={(e) => set({ sla: { ...w.sla, p99_ttft_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
          <div><Label help="Typical time between two generated tokens (median); 150 ms is about reading speed. Empty = no limit.">Time per generated word (ms)</Label>
            <input type="number" value={w.sla.p50_tpot_ms ?? ""} onChange={(e) => set({ sla: { ...w.sla, p50_tpot_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
          <div><Label help="Optional overall limit per request, 99th percentile. Empty = no limit.">Whole answer within (ms)</Label>
            <input type="number" value={w.sla.p99_e2e_ms ?? ""} onChange={(e) => set({ sla: { ...w.sla, p99_e2e_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
        </div>
      ) : (
        <div className="tiles">
          {Object.entries(cat.presets).map(([k, p]) => <Tile key={k} selected={w.preset === k} onClick={() => set({ preset: k })} title={p.label} desc={`${p.help} (99 of 100 decisions within ${fmt.n(p.p99_e2e_ms)} ms.)`} />)}
        </div>
      )}
    </div>
  );
}

function StepReview({ w, set, onStarted }: SP & { onStarted: (id: string) => void }) {
  const cat = useCatalog()!;
  const [plan, setPlan] = useState<Plan | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const body = useMemo(() => JSON.stringify(w), [w]);
  useEffect(() => { setPlan(null); setErr(null); api.plan(JSON.parse(body)).then(setPlan).catch((e) => setErr(e.message)); }, [body]);
  const start = async () => {
    setStarting(true);
    try { const r = await api.start(w); sessionStorage.removeItem("wizard"); onStarted(r.run_id); }
    catch (e: any) { setErr(e.message); } finally { setStarting(false); }
  };
  const hwLabel = (id: string, it: string) => { const h = cat.hardware.find((x) => x.id === id); return h ? `${h.accelerator.toUpperCase()} · ${it} · ${h.software.join(" + ")}` : id; };
  return (
    <div>
      <h2>7 · Review and start</h2>
      <div className="grid c2 mb">
        <div className="card soft">
          <h3>What will run</h3>
          <ul className="muted" style={{ lineHeight: 1.8, paddingLeft: 18 }}>
            <li><b>{cat.use_cases[w.use_case].name}</b>{w.use_case === "standard" ? ` · ${cat.prompt_sets[w.prompt_set].label}, ${w.answer_tokens}-token answers` : ` · ${w.workloads.map((x) => cat.workloads[x].label).join("; ")} · ${w.test_items} documents in the accuracy check`}</li>
            <li>Models: <b>{w.models.map((id) => cat.models.find((m) => m.id === id)?.name).join(", ")}</b></li>
            <li>Hardware: {w.gpu && <Badge kind="gpu">{hwLabel(w.gpu.hardware, w.gpu.instance_type)}</Badge>} {w.cpus.map((c) => <Badge key={c.hardware} kind="cpu">{hwLabel(c.hardware, c.instance_type)}</Badge>)} · {w.parallel_targets ? "in parallel" : "one after another"} · {cat.purchase[w.purchase].label}</li>
            <li>{cat.request_modes[w.request_mode].label}{w.request_mode !== "single" ? ` at ${w.concurrency.join(", ")} concurrent` : ""}{w.request_mode === "batch" ? `, batches of ${w.batch_sizes.join(", ")}` : ""}</li>
            <li>{w.depth === "custom" ? `${w.requests} requests per level, ${w.passes} passes` : cat.depths[w.depth].label}</li>
            <li>Targets: {w.use_case === "dspm" ? cat.presets[w.preset].label : Object.entries(w.sla).filter(([, v]) => v).map(([k, v]) => `${{ p99_ttft_ms: "first word ≤", p50_tpot_ms: "per word ≤", p99_e2e_ms: "whole answer ≤" }[k as string] ?? k} ${v} ms`).join(", ") || "none"}</li>
          </ul>
          <Label help="A short label for this run; it appears in the run list and in file names.">Run name</Label>
          <input value={w.name} placeholder={`${w.use_case}-${new Date().toISOString().slice(5, 10)}`} onChange={(e) => set({ name: e.target.value })} />
        </div>
        <div className="card soft">
          <h3>Estimate</h3>
          {!plan && !err && <Spinner label="Validating and estimating…" />}
          {plan?.estimate && (
            <div className="grid c3">
              <div className="metric"><div className="k">Wall-clock time</div><div className="v">{plan.estimate.wall_clock_hours.toFixed(1)} h</div><div className="d muted">assumed speeds until measured</div></div>
              <div className="metric"><div className="k">Cost</div><div className="v">{fmt.usd(plan.estimate.usd.total)}</div><div className="d muted">On-Demand rates; Spot is usually 30–70% less</div></div>
              <div className="metric"><div className="k">Grade</div><div className="v" style={{ fontSize: 18 }}>{plan.grade === "full" ? "Decision-grade" : "Quick check"}</div><div className="d muted">{plan.grade === "full" ? "≥200 requests × 3 passes" : "labelled SMOKE"}</div></div>
            </div>
          )}
          {plan && (
            <table className="tbl mt">
              <thead><tr><th>Model</th><th>Hardware</th><th>Format</th><th className="num">Measurements</th></tr></thead>
              <tbody>{plan.series.map((s, i) => <tr key={i}><td>{s.model}</td><td><Badge kind={s.accelerator}>{s.accelerator.toUpperCase()}</Badge> {s.instance_type} · {s.engine}</td><td>{s.precision}</td><td className="num">{s.measurements}</td></tr>)}</tbody>
            </table>
          )}
        </div>
      </div>
      {plan?.pairing.map((p, i) => <Note key={i} kind={p.fair ? "ok" : "warn"}><b>{p.fair ? "Fair comparison" : "Not a fair comparison"}</b> — {p.reason}.</Note>)}
      {plan?.warnings.map((x, i) => <Note key={"w" + i} kind="warn">{x}</Note>)}
      {plan?.problems.map((x, i) => <Note key={"p" + i} kind="bad">{x}</Note>)}
      {plan && !plan.ok && plan.problems.some((p) => p.startsWith("Unfair pairing")) && (
        <label className="row small mt"><input type="checkbox" checked={w.allow_unfair} onChange={(e) => set({ allow_unfair: e.target.checked })} /> I understand the sizes are not comparable; start anyway</label>
      )}
      {err && <Note kind="bad">{err}</Note>}
      <div className="row mt">
        <button className="btn primary" disabled={!plan?.ok || starting} onClick={start}>{starting ? <Spinner label="Starting…" /> : "Start the benchmark"}</button>
        {plan && <details><summary className="small">Advanced: the run definition the command line uses</summary><pre>{plan.yaml}</pre></details>}
      </div>
    </div>
  );
}
