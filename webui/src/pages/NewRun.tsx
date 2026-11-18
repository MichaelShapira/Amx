import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import Markdown from "react-markdown";
import { api, defaultWizard, fmt, type AiJob, type AiProposal, type AiStep, type Catalog, type Hardware, type ModelInfo, type Plan, type Side, type UseCase, type Wizard } from "../api";
import { useCatalog } from "../App";
import DataPreview from "../components/DataPreview";
import { Badge, Chips, Help, Label, Note, Spinner, Tile, Toggle, instanceLine } from "../components/ui";

const STEPS = ["Use case", "Test data", "Depth", "Models", "Hardware", "Load & targets", "Review"];

export default function NewRun() {
  const cat = useCatalog();
  const nav = useNavigate();
  const loc = useLocation();
  const [step, setStep] = useState<number>(() => Math.min(STEPS.length - 1, Math.max(0, Number((loc.state as any)?.step ?? 0) || 0)));
  const [w, setW] = useState<Wizard>(() => {
    try { const s = sessionStorage.getItem("wizard"); return s ? { ...defaultWizard, ...JSON.parse(s) } : defaultWizard; } catch { return defaultWizard; }
  });
  useEffect(() => { sessionStorage.setItem("wizard", JSON.stringify(w)); }, [w]);
  const set = (patch: Partial<Wizard>) => setW((p) => ({ ...p, ...patch }));
  useEffect(() => {
    if (!cat) return;
    const known = new Set(cat.hardware.map((h) => h.id));
    const seen = new Set<string>();
    const cpus = w.cpus.filter((c) => known.has(c.hardware) && !seen.has(c.hardware) && seen.add(c.hardware));
    const gpu = w.gpu && known.has(w.gpu.hardware) ? w.gpu : null;
    if (cpus.length !== w.cpus.length || gpu !== w.gpu) set({ cpus, gpu });
  }, [cat]);  // eslint-disable-line react-hooks/exhaustive-deps

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
      {w.copied_from && (
        <Note kind="ok"><b>Copied from run {w.copied_from}.</b> Every step is prefilled with that run's settings and nothing has started: click any step above to change it, then press Start on the last step.
          {w.recommendation && <div className="mt small"><b>Recommended sizes applied.</b> {w.recommendation}</div>}
          <button className="btn sm" style={{ marginLeft: 8 }} onClick={() => set({ copied_from: null, recommendation: null })}>Dismiss</button></Note>
      )}
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
    const defaults = uc === "dspm" ? { concurrency: [1, 8, 16], sla: {} }
      : uc === "translation" ? { concurrency: [1, 4, 16], sla: { p99_e2e_ms: 10000, p99_ttft_ms: 2000 } }
      : { concurrency: [1, 4, 16], sla: { p99_ttft_ms: 2000, p50_tpot_ms: 150 } };
    set({ use_case: uc, models: [], ...defaults });
  };
  return (
    <div>
      <h2>1 · What do you want to size hardware for?</h2>
      <p className="muted">Each use case uses its own test data and success criteria; a run covers one.</p>
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
      ) : w.use_case === "translation" ? (
        <>
          <p className="muted">A synthetic parallel corpus: every item is written in English, German and Spanish from the same template, so the reference translation is known and nothing is real. Pick the directions and the size of the texts; each direction is measured separately.</p>
          <Label help="Each direction is its own measurement: the source language decides how much the model reads, the target language how much it writes (German and Spanish run about 10-20% longer than English).">Directions (pick one or more)</Label>
          <div className="tiles mb">
            {Object.entries(cat.directions).map(([k, p]) => (
              <Tile key={k} selected={w.directions.includes(k)} onClick={() => set({ directions: w.directions.includes(k) ? w.directions.filter((x) => x !== k) : Object.keys(cat.directions).filter((d) => d === k || w.directions.includes(d)) })} title={p.label} desc={p.help} />
            ))}
          </div>
          <Label help="How long each text is. Answers end where the model ends them (no fixed length), with a token limit set per size so a complete translation always fits.">Text size</Label>
          <div className="tiles mb">
            {Object.entries(cat.text_sizes).map(([k, p]) => <Tile key={k} selected={w.text_size === k} onClick={() => set({ text_size: k })} title={p.label} desc={`${p.help} Answer limit ${p.answer_tokens} tokens.`} />)}
          </div>
          <Label help="How many of the 200 items per direction each option translates once for the quality figures (chrF against the reference and against the GPU's translations). Every option translates the same items.">Items in the quality pass (per direction)</Label>
          <Chips options={[50, 100, 200]} value={[w.translation_items]} onChange={(v) => set({ translation_items: v.find((x) => x !== w.translation_items) ?? w.translation_items })} />
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
      <h3>Preview <span className="muted small">— real examples of what the model will see</span></h3>
      <DataPreview useCase={w.use_case} promptSet={w.use_case === "standard" ? w.prompt_set : w.use_case === "translation" ? w.text_size : undefined} workloads={w.use_case === "dspm" ? w.workloads : undefined} directions={w.use_case === "translation" ? w.directions : undefined} compact model={w.models[0]} />
    </div>
  );
}

function StepDepth({ w, set }: SP) {
  const cat = useCatalog()!;
  return (
    <div>
      <h2>3 · Quick check or full test?</h2>
      <p className="muted">A measurement is one load setting (for example 10 requests at a time) on one hardware option. More requests per measurement give steadier percentiles; repeating the whole test shows whether a result holds.</p>
      <div className="tiles mb">
        {Object.entries(cat.depths).map(([k, d]) => (
          <Tile key={k} selected={w.depth === k} onClick={() => set({ depth: k as any })} title={d.label} desc={d.help} badge={<Badge kind={k === "full" ? "ok" : "warn"}>{k === "full" ? "decision-grade" : "indicative only"}</Badge>} />
        ))}
        <Tile selected={w.depth === "custom"} onClick={() => set({ depth: "custom", requests: w.requests ?? 200, passes: w.passes ?? 3 })} title="Custom" desc="Choose the number of requests per measurement and how many times to repeat the test.">
          {w.depth === "custom" && (
            <div className="row mt" onClick={(e) => e.stopPropagation()}>
              <label className="small">Requests per measurement <input type="number" min={1} max={100000} value={w.requests ?? 200} onChange={(e) => set({ requests: Number(e.target.value) })} style={{ width: 90 }} /></label>
              <label className="small">Repeat the test <input type="number" min={1} max={10} value={w.passes ?? 3} onChange={(e) => set({ passes: Number(e.target.value) })} style={{ width: 70 }} /></label>
            </div>
          )}
        </Tile>
      </div>
      <Note>{w.depth === "full" || (w.depth === "custom" && (w.requests ?? 0) >= 200 && (w.passes ?? 0) >= 3) ? "This run will be decision-grade." : "This run will be a quick check: the report marks every number as indicative only (SMOKE); it must not drive a buying decision."}</Note>
    </div>
  );
}

function StepModels({ w, set }: SP) {
  const cat = useCatalog()!;
  const models = cat.models.filter((m) => m.use_cases.includes(w.use_case));
  // One model per benchmark: the hardware step sizes both sides for it (a 0.4B classifier and a
  // 4B chat model need different machines), and the results stay one question, one answer.
  // Targets follow the kind of model: a chat model is judged on the first word and each generated word,
  // a classifier (no words generated) on the whole decision only.
  const pick = (id: string) => {
    const m = cat.models.find((x) => x.id === id);
    const generative = !m || m.type === "decoder" || m.type === "encoder-decoder";
    set({ models: [id], ...(w.use_case === "standard" ? { sla: generative ? { p99_ttft_ms: 2000, p50_tpot_ms: 150 } : { p99_e2e_ms: 1000 } } : w.use_case === "translation" ? { sla: { p99_e2e_ms: 10000, p99_ttft_ms: 2000 } } : {}) });
  };
  return (
    <div>
      <h2>4 · Which model?</h2>
      <p className="muted">Pick one. The next step sizes the GPU and the CPU for exactly this model: its kind (chat model or classifier) and size decide the memory, the number of cores and the fair instance pair. Run another benchmark for another model; results can be compared side by side afterwards. The license says whether a commercial decision may rest on the result.</p>
      <div className="tiles">
        {models.map((m) => (
          <Tile key={m.id} selected={w.models[0] === m.id} onClick={() => pick(m.id)} title={m.name}
            badge={<><Badge>{m.params_b}B</Badge><Badge kind={m.commercial_ok ? "ok" : "warn"}>{m.commercial_ok ? m.license : "research license"}</Badge>{m.user_added && <Badge kind="accent">added in Setup · not yet run</Badge>}</>}
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
  const [ai, setAi] = useState<AiJob | null>(null);
  const [aiErr, setAiErr] = useState<string | null>(null);
  const chosenModels = cat.models.filter((m) => w.models.includes(m.id));
  const model = chosenModels[0];
  // Poll the proposal job; when it ends with a valid proposal and nothing was sized for this model
  // yet, apply it (the page's pre-selection). Later proposals wait for "Use this proposal".
  useEffect(() => {
    if (!ai || ai.status !== "running") return;
    const t = setInterval(() => api.aiProposeStatus(ai.id).then((j) => {
      setAi(j);
      if (j.status === "done" && j.result?.proposal && sizeKey && w.sized_for !== sizeKey) { applyProposal(j.result.proposal); }
      else if (j.status !== "running" && sizeKey && w.sized_for !== sizeKey) set({ sized_for: sizeKey });
    }).catch((e) => { setAiErr(e.message); setAi(null); }), 2000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ai]);

  // A number format the option's serving software serves but whose CPU feature the option's
  // instances do not expose (FP16 needs AMX-FP16; no AWS Intel instance has it) is blocked here,
  // at selection, with the reason: the same facts the plan and the pre-checks use.
  const flagBlock = (hw: Hardware, spec: ModelInfo["precisions"][string]) =>
    hw.accelerator === "cpu" && spec.requires_cpu_flag && !hw.cpu_features.includes(spec.requires_cpu_flag) ? spec.requires_cpu_flag : null;
  const precsFor = (hwId: string) => {
    const hw = cat.hardware.find((h) => h.id === hwId)!;
    const set_ = new Set<string>();
    for (const m of chosenModels) for (const [p, spec] of Object.entries(m.precisions)) if (spec.engines.some((e) => hw.engines.includes(e)) && !spec.quality_only && !flagBlock(hw, spec)) set_.add(p);
    return [...set_];
  };
  const blockedFor = (hwId: string) => {
    const hw = cat.hardware.find((h) => h.id === hwId)!;
    const out = new Map<string, string>();
    for (const m of chosenModels) for (const [p, spec] of Object.entries(m.precisions)) {
      const flag = spec.engines.some((e) => hw.engines.includes(e)) && !spec.quality_only ? flagBlock(hw, spec) : null;
      if (flag && !out.has(p)) out.set(p, `${cat.precisions[p]?.label ?? p} ${cat.cpu_flag_notes[flag] ?? `needs the ${flag} CPU feature, which this option's instances do not have.`}`);
    }
    return [...out].map(([option, reason]) => ({ option, reason }));
  };
  // The default number format for a hardware option: BF16 when its serving software can run the
  // model in BF16 (OpenVINO's own BF16 cell counts), otherwise the first format it can.
  const defaultPrecs = (hwId: string) => { const ok = precsFor(hwId); const bf16 = ["bf16", "bf16-ov"].find((p) => ok.includes(p)); return bf16 ? [bf16] : ok.slice(0, 1); };
  // Number formats carried over from another model may not exist for this one: keep the valid ones, else the default.
  const fixPrecs = (hwId: string, cur: string[] | undefined) => { const ok = precsFor(hwId); const kept = (cur ?? []).filter((p) => ok.includes(p)); return kept.length ? kept : defaultPrecs(hwId); };
  const applyProposal = (p: AiProposal) => {
    const gpuId = w.gpu?.hardware ?? gpuHw[0].id;
    const firstOk = cpuHw.find((h) => precsFor(h.id).length > 0) ?? cpuHw[0];      // vLLM cannot serve every model
    const base = w.cpus.length ? w.cpus : [{ hardware: firstOk.id, instance_type: p.cpu.instance_type, precisions: [] as string[], alternates: firstOk.alternates }];
    set({ sized_for: sizeKey ?? undefined,
      gpu: { hardware: gpuId, instance_type: p.gpu.instance_type, precisions: fixPrecs(gpuId, w.gpu?.precisions), alternates: p.gpu.alternates },
      cpus: base.map((c) => ({ ...c, instance_type: p.cpu.instance_type, precisions: fixPrecs(c.hardware, c.precisions) })) });
  };
  const aiPropose = async () => {
    setAiErr(null);
    try { const { job_id } = await api.aiProposeStart(w); setAi(await api.aiProposeStatus(job_id)); }
    catch (e) { setAiErr((e as Error).message); }
  };
  // The first time this step is shown for a model (or after the model or number format changed),
  // ask the reasoning model for the pair to compare. Manual changes afterwards are kept until the model changes.
  const sizeKey = model ? `${model.id}|${w.precision}` : null;
  useEffect(() => {
    if (!model || !sizeKey || w.sized_for === sizeKey || ai?.status === "running") return;
    // A starting point before (or without) the proposal: the GPU option and the first CPU option
    // whose serving software can run this model, each at its default size.
    if (!w.gpu || w.cpus.length === 0) {
      const firstOk = cpuHw.find((h) => precsFor(h.id).length > 0) ?? cpuHw[0];
      set({ gpu: w.gpu ?? { hardware: gpuHw[0].id, instance_type: gpuHw[0].default_instance_type, precisions: defaultPrecs(gpuHw[0].id), alternates: gpuHw[0].alternates },
            cpus: w.cpus.length ? w.cpus : [{ hardware: firstOk.id, instance_type: firstOk.default_instance_type, precisions: defaultPrecs(firstOk.id), alternates: firstOk.alternates }] });
    }
    void aiPropose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sizeKey, w.sized_for]);
  const toggleGpu = (hw: typeof gpuHw[number]) => set({ gpu: w.gpu?.hardware === hw.id ? null : { hardware: hw.id, instance_type: w.gpu?.instance_type ?? hw.default_instance_type, precisions: defaultPrecs(hw.id), alternates: w.gpu?.alternates ?? hw.alternates } });
  const toggleCpu = (hw: typeof cpuHw[number]) => {
    const on = w.cpus.find((c) => c.hardware === hw.id);
    set({ cpus: on ? w.cpus.filter((c) => c.hardware !== hw.id) : [...w.cpus, { hardware: hw.id, instance_type: w.cpus[0]?.instance_type ?? hw.default_instance_type, precisions: defaultPrecs(hw.id), alternates: hw.alternates }] });
  };
  const setSide = (hwId: string, patch: Partial<Side>) => {
    if (w.gpu?.hardware === hwId) set({ gpu: { ...w.gpu, ...patch } });
    else set({ cpus: w.cpus.map((c) => (c.hardware === hwId ? { ...c, ...patch } : c)) });
  };
  const gpuTypes = Object.values(cat.instances).filter((s) => s.accelerator === "gpu").sort((a, b) => a.usd_per_hour - b.usd_per_hour);
  const cpuTypes = Object.values(cat.instances).filter((s) => s.accelerator === "cpu").sort((a, b) => a.usd_per_hour - b.usd_per_hour);
  const proposal = ai?.result?.proposal ?? null;
  const applied = !!proposal && w.gpu?.instance_type === proposal.gpu.instance_type && w.cpus.length > 0 && w.cpus.every((c) => c.instance_type === proposal.cpu.instance_type);

  const sideBlock = (hw: typeof cat.hardware[number], side: Side | undefined, kind: "gpu" | "cpu") => (
    <div key={hw.id} className={`tile ${kind} ${side ? "selected" : ""}`} style={{ cursor: "default" }}>
      <span className="check" onClick={() => (kind === "gpu" ? toggleGpu(hw as any) : toggleCpu(hw as any))} style={{ cursor: "pointer" }}>{side ? "✓" : ""}</span>
      <div className="t" onClick={() => (kind === "gpu" ? toggleGpu(hw as any) : toggleCpu(hw as any))} style={{ cursor: "pointer" }}>
        <Badge kind={kind}>{kind.toUpperCase()}</Badge> {kind === "gpu" ? "NVIDIA" : "Intel Xeon with AMX"} · {hw.software.join(" + ")}
        <Help text={hw.software_help} />
      </div>
      {side && (
        <div className="mt">
          <Label help={kind === "gpu" ? "The AWS instance size for the GPU side. The proposal below picks one for this model; change it here if you know better. The report prices results with the On-Demand rate of the size that ran." : "The AWS instance size for this option. The proposal below picks one priced close to the GPU; change it here if you know better."}>Instance size</Label>
          <select value={side.instance_type} onChange={(e) => setSide(hw.id, { instance_type: e.target.value })} style={{ width: "100%" }}>
            {(kind === "gpu" ? gpuTypes : cpuTypes).map((s) => <option key={s.type} value={s.type}>{s.type} · {instanceLine(s, true)}</option>)}
          </select>
          <Label help="Number formats to test on this hardware. BF16 is the like-for-like baseline; compressed formats (INT8, INT4) are faster and cheaper on CPUs and the report shows what they cost in answer quality.">Number formats</Label>
          {precsFor(hw.id).length === 0 && <Note kind="warn">This serving software cannot run {model?.name ?? "the chosen model"} at all. Untick it, or pick another model.</Note>}
          <Chips options={precsFor(hw.id)} value={side.precisions.filter((p) => precsFor(hw.id).includes(p))} onChange={(v) => setSide(hw.id, { precisions: v.length ? v : defaultPrecs(hw.id) })} labels={(p) => cat.precisions[p]?.label ?? p} blocked={blockedFor(hw.id)} />
          {blockedFor(hw.id).map((b) => <div key={b.option} className="muted small" style={{ marginTop: 6 }}>{b.reason}</div>)}
        </div>
      )}
    </div>
  );

  return (
    <div>
      <h2>5 · Which hardware to compare?</h2>
      <p className="muted">The reasoning model (Setup → Reasoning model) proposes the GPU size and the AMX CPU size to compare for <b>{model?.name ?? "the chosen model"}</b>, checking the model, the hardware and today's prices on the web, and explains why in plain words. The proposal is applied for you the first time; change either size, or tick more CPU options (different serving software is a different option; each becomes one test instance).</p>
      {model?.user_added && (
        <Note kind="warn"><b>{model.name} was added in Setup.</b> Nothing has run it here yet, so which serving software really loads it is expected, not proven. The first benchmark is the real test; if a hardware option cannot start the model, the run page says exactly why and the other options still finish.</Note>
      )}
      {aiErr && <Note kind="bad">{aiErr}</Note>}
      {ai && <AiProposalView job={ai} applied={applied} onAgain={aiPropose} onApply={applyProposal} />}
      <div className="grid c2 mt">
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
      <Label help="Run every hardware option at the same time (fastest; one instance per option is up at once) or one after another (fewer instances running at any moment; the run takes longer). Results are the same either way.">How to run the hardware options</Label>
      <Toggle value={w.parallel_targets ? "par" : "seq"} onChange={(v) => set({ parallel_targets: v === "par" })} options={[{ value: "par", label: "All in parallel" }, { value: "seq", label: "One after another" }]} />
      <div className="mt">
        <Label help="Spot: spare capacity at a discount that AWS may reclaim with two minutes' notice; the run then launches a replacement and re-runs only the unfinished measurements. On-Demand: guaranteed for the run, full price. Cost columns always use On-Demand prices so results stay comparable.">How to buy the test instances</Label>
        <Toggle value={w.purchase} onChange={(v) => set({ purchase: v })} options={[{ value: "spot", label: "Spot (cheaper, may be interrupted)" }, { value: "on-demand", label: "On-Demand (guaranteed)" }]} />
        {w.purchase === "spot" && (
          <div className="row small mt" style={{ alignItems: "center", gap: 8 }}>
            <label className="row small" style={{ gap: 6 }}><input type="checkbox" checked={w.spot_fallback} onChange={(e) => set({ spot_fallback: e.target.checked })} /> Fall back to On-Demand when Spot has had no capacity for</label>
            <input type="number" min={1} max={720} value={w.spot_wait_minutes} disabled={!w.spot_fallback} onChange={(e) => set({ spot_wait_minutes: Math.max(1, Number(e.target.value) || 1) })} style={{ width: 70 }} /> minutes
            <Help text="Spot is tried first. If AWS keeps answering 'no capacity', the run switches that hardware option to On-Demand (full price, guaranteed) after this many minutes. Untick to stay on Spot only." />
          </div>
        )}
        <div className="row small mt" style={{ alignItems: "center", gap: 8 }}>
          <span>Give up on a hardware option after</span>
          <input type="number" min={1} max={720} value={w.launch_attempts} onChange={(e) => set({ launch_attempts: Math.max(1, Number(e.target.value) || 1) })} style={{ width: 70 }} />
          <span>failed launch attempts (about one a minute, so roughly {w.launch_attempts} minutes)</span>
          <Help text="AWS sometimes has no machines of the requested size in the zone, especially GPUs. The run asks again about once a minute, trying the alternate sizes too. After this many refusals it gives up on that hardware option, marks it 'Did not run — no capacity' with the full reason, and finishes the other options. A higher number waits longer for capacity; a lower one fails fast." />
        </div>
        {w.launch_attempts < 10 && <Note kind="warn">Fewer than 10 attempts: GPU sizes are often refused several times in a row before one is granted. The GPU side may not run at all.</Note>}
      </div>
    </div>
  );
}

function AiProposalView({ job, applied, onAgain, onApply }: { job: AiJob; applied: boolean; onAgain: () => void; onApply: (p: AiProposal) => void }) {
  const cat = useCatalog()!;
  const r = job.result;
  const p = r?.proposal ?? null;
  const stepLine = (s: AiStep) => s.kind === "repair" ? `Asked again for a shorter, JSON-only answer (${s.text})` : `Searched: “${s.text}” (${s.n ?? 0} results${s.error ? `, ${s.error}` : ""})`;
  const spec = (t: string) => cat.instances[t];
  const ratio = p && spec(p.gpu.instance_type) && spec(p.cpu.instance_type) ? spec(p.gpu.instance_type).usd_per_hour / spec(p.cpu.instance_type).usd_per_hour : null;
  const confKind = p?.confidence === "high" ? "ok" : p?.confidence === "low" ? "warn" : "accent";
  return (
    <div className="card mt" style={{ borderColor: "var(--accent)" }}>
      <div className="row between">
        <h3 style={{ margin: 0 }}>Proposed pair for this test <Help text="The reasoning model chosen in Setup (Claude Opus by default) receives the model, the test's shape, every allowed instance size with its catalog specs and prices, and this tool's fairness rule (hourly prices within half to double of each other). It searches the web for what it is not sure of (the model's size and architecture, GPU memory, current prices, whether the serving software supports the model) and proposes one GPU size and one CPU size, with reasons in plain words. Everything it reads on the web is treated as data, not instructions, and the proposal is only offered when both sizes are on this tool's allowed lists." /></h3>
        <div className="row small muted">
          {job.model && <span>{job.model.name} · Bedrock</span>}
          {job.status !== "running" && <button className="btn sm" onClick={onAgain}>Propose again</button>}
        </div>
      </div>
      {job.status === "running" && (
        <div className="mt">
          <Spinner label={`Working it out with web search… ${job.steps.length} search${job.steps.length === 1 ? "" : "es"} so far, ${Math.round(job.seconds)} s. This usually takes one to three minutes; you can change the sizes below meanwhile.`} />
          <ul className="small muted mt">{job.steps.map((s, i) => <li key={i}>{stepLine(s)}</li>)}</ul>
        </div>
      )}
      {job.status === "error" && <Note kind="bad"><b>The proposal failed.</b> {job.error} Pick the sizes by hand below, or press Propose again.</Note>}
      {r && !p && (
        <div className="mt">
          <Note kind="warn"><b>No proposal this time:</b> {r.problem}. Pick the sizes by hand below, or press Propose again. What the model wrote:</Note>
          {r.markdown && <div className="prose mt"><Markdown>{r.markdown}</Markdown></div>}
        </div>
      )}
      {r && p && (
        <>
          <div className="grid c2 mt">
            <div className="tile gpu selected" style={{ cursor: "default" }}>
              <div className="t" style={{ fontSize: 18 }}><Badge kind="gpu">GPU</Badge> {p.gpu.instance_type}</div>
              {spec(p.gpu.instance_type) && <div className="d">{instanceLine(spec(p.gpu.instance_type))}</div>}
              <div className="small mt">{p.gpu.why}</div>
              {p.gpu.alternates.length > 0 && <div className="tiny mt">If AWS has none available, the run tries {p.gpu.alternates.join(" and ")}.</div>}
            </div>
            <div className="tile cpu selected" style={{ cursor: "default" }}>
              <div className="t" style={{ fontSize: 18 }}><Badge kind="cpu">CPU</Badge> {p.cpu.instance_type}</div>
              {spec(p.cpu.instance_type) && <div className="d">{instanceLine(spec(p.cpu.instance_type))}</div>}
              <div className="small mt">{p.cpu.why}</div>
            </div>
          </div>
          <Note kind="ok">
            <b>Why this is a fair comparison.</b> {p.summary}
            {ratio != null && <span> The GPU costs <b>{ratio.toFixed(2)}x</b> as much per hour as the CPU{ratio >= cat.fair_ratio[0] && ratio <= cat.fair_ratio[1] ? ", inside this tool's fair band" : ", OUTSIDE this tool's fair band (half to double)"}.</span>}
          </Note>
          <div className="row mt">
            <button className="btn primary" disabled={applied} onClick={() => onApply(p)}>{applied ? "✓ Selected below" : "Use this proposal"}</button>
            <Badge kind={confKind}>confidence: {p.confidence}</Badge>
          </div>
          {p.caveats.length > 0 && (
            <div className="mt"><div className="label">Good to know</div><ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>{p.caveats.map((x, i) => <li key={i}>{x}</li>)}</ul></div>
          )}
          <details className="mt"><summary className="small">What was checked ({p.checked.length} facts, {r.tool_calls} web searches, {r.seconds} s)</summary>
            <ul className="small muted mt">{p.checked.map((x, i) => <li key={i}>{x}</li>)}</ul>
            <div className="label mt">Searches</div>
            <ul className="small muted">{r.steps.map((s, i) => <li key={i}>{stepLine(s)}</li>)}</ul>
            {r.sources.length > 0 && <><div className="label mt">Sources</div><ul className="small">{r.sources.map((s) => <li key={s.url}><a href={s.url} target="_blank" rel="noreferrer">{s.title}</a></li>)}</ul></>}
          </details>
          <div className="tiny muted mt">Written by {r.model.name} ({r.model.call_id}) at {r.generated_at}. Advice from a language model grounded in the pages it lists; check a source before acting on a number.</div>
        </>
      )}
    </div>
  );
}

function StepLoad({ w, set }: SP) {
  const cat = useCatalog()!;
  const chosen = cat.models.filter((m) => w.models.includes(m.id));
  const batchOk = chosen.length > 0 && chosen.every((m) => m.batch_capable);
  const generative = chosen.length === 0 || chosen.some((m) => m.type === "decoder" || m.type === "encoder-decoder");
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
          <Label help="How many requests are in flight at the same time on each hardware option. Each number you pick is measured separately. Waiting time and throughput both rise with more concurrent requests; the verdict looks for the highest number where the machine still meets your targets.">How many requests at a time? (pick one or more)</Label>
          <Chips options={levels} value={w.concurrency} onChange={(v) => set({ concurrency: v.length ? v : [1] })} labels={(c) => cat.concurrency_hints[String(c)] ? `${c} · ${cat.concurrency_hints[String(c)]}` : String(c)} />
        </>
      )}
      {w.request_mode === "batch" && (
        <div className="mt">
          <Label help="How many inputs go into one request. Classifiers score a whole batch in one go; larger batches raise throughput and waiting time together.">Inputs per request (pick one or more)</Label>
          <Chips options={[1, 4, 8, 16, 32, 64]} value={w.batch_sizes} onChange={(v) => set({ batch_sizes: v.length ? v : [8] })} />
        </div>
      )}
      <div className="hr" />
      <h3>What counts as acceptable? <Help text="The limits a hardware option must meet to count. The report marks every measurement that meets them, then picks the fastest GPU and the fastest CPU among those. Capacity (can the CPU side match the GPU's throughput with a reasonable number of machines) decides the verdict; cost is shown alongside." /></h3>
      {w.use_case === "standard" && !generative ? (
        <div className="grid c3">
          <div><Label help="A classifier gives its whole answer at once, so this is the one time limit that applies: 99 of 100 decisions within this many milliseconds. 1,000 ms suits a real-time check; empty = no limit.">Whole decision within (ms)</Label>
            <input type="number" value={w.sla.p99_e2e_ms ?? ""} onChange={(e) => set({ sla: { p99_e2e_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
          <div className="small muted" style={{ gridColumn: "span 2", alignSelf: "end" }}>{chosen[0]?.name ?? "This model"} generates no words, so there is no "first word" or "per word" target to set.</div>
        </div>
      ) : w.use_case === "translation" ? (
        <div className="grid c3">
          <div><Label help="The longest acceptable time for the whole translation, for 99 of 100 requests. A short message should come back in a few seconds; a paragraph in ten or so. Empty = no limit.">Whole translation within (ms)</Label>
            <input type="number" value={w.sla.p99_e2e_ms ?? ""} onChange={(e) => set({ sla: { ...w.sla, p99_e2e_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
          <div><Label help="The longest acceptable wait before the translation starts appearing, for 99 of 100 requests (matters when the translation is streamed to a reader). Empty = no limit.">First word within (ms)</Label>
            <input type="number" value={w.sla.p99_ttft_ms ?? ""} onChange={(e) => set({ sla: { ...w.sla, p99_ttft_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
          <div><Label help="Typical time between two generated tokens (median). Empty = no limit.">Time per generated word (ms)</Label>
            <input type="number" value={w.sla.p50_tpot_ms ?? ""} onChange={(e) => set({ sla: { ...w.sla, p50_tpot_ms: e.target.value ? Number(e.target.value) : undefined } })} /></div>
        </div>
      ) : w.use_case === "standard" ? (
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

function list(xs: (number | string)[]): string {
  const a = xs.map(String);
  return a.length <= 1 ? a.join("") : a.slice(0, -1).join(", ") + " and " + a[a.length - 1];
}

/** "one request at a time", "1, 4 and 16 requests at a time", "batches of 8 and 32 inputs, 4 requests at a time". */
export function loadSentence(w: Wizard): string {
  if (w.request_mode === "single") return "one request at a time";
  const conc = w.concurrency.length === 1 ? `${w.concurrency[0]} request${w.concurrency[0] === 1 ? "" : "s"} at a time` : `${list(w.concurrency)} requests at a time (each measured separately)`;
  if (w.request_mode === "batch") return `batches of ${list(w.batch_sizes)} inputs per request, ${conc}`;
  return conc;
}

export function depthSentence(w: Wizard, cat: Catalog): string {
  const req = w.depth === "custom" ? w.requests ?? 200 : cat.depths[w.depth].requests;
  const passes = w.depth === "custom" ? w.passes ?? 3 : cat.depths[w.depth].passes;
  const grade = req >= 200 && passes >= 3 ? "decision-grade" : "a quick check, indicative only";
  return `${req} requests for each measurement, ${passes === 1 ? "run once" : `repeated ${passes} times`}: ${grade}`;
}

function StepReview({ w, set, onStarted }: SP & { onStarted: (id: string) => void }) {
  const cat = useCatalog()!;
  const [plan, setPlan] = useState<Plan | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  // The name does not change the plan; typing it must not re-validate and flicker the estimate.
  const body = useMemo(() => JSON.stringify({ ...w, name: "" }), [w]);
  useEffect(() => { setPlan(null); setErr(null); api.plan(JSON.parse(body)).then(setPlan).catch((e) => setErr(e.message)); }, [body]);
  const [slug, setSlug] = useState<{ slug: string; run_id: string } | null>(null);
  useEffect(() => { const t = setTimeout(() => api.slug(w.name, w.use_case).then(setSlug).catch(() => setSlug(null)), 250); return () => clearTimeout(t); }, [w.name, w.use_case]);
  const start = async () => {
    setStarting(true);
    try { const r = await api.start(w); sessionStorage.removeItem("wizard"); onStarted(r.run_id); }
    catch (e: any) { setErr(e.message); } finally { setStarting(false); }
  };
  const hwLabel = (id: string, it: string) => { const h = cat.hardware.find((x) => x.id === id); return h ? `${h.accelerator.toUpperCase()} · ${it} · ${h.software.join(" + ")}` : id; };
  return (
    <div>
      <h2>7 · Review and start</h2>
      <div className="card soft mb">
        <Label help="A short name for this benchmark, for you. It becomes the first part of the run id (lowercase letters, digits and dashes; spaces become dashes), followed by the start time. Leave it empty for an automatic name.">Name this benchmark</Label>
        <div className="row" style={{ alignItems: "center", gap: 12 }}>
          <input value={w.name} placeholder="e.g. roberta-vs-l4 or gemma-first-look" onChange={(e) => set({ name: e.target.value })} style={{ maxWidth: 420 }} autoFocus />
          <span className="small muted">Run id will be <code>{slug?.run_id ?? "…"}</code></span>
        </div>
      </div>
      <div className="grid c2 mb">
        <div className="card soft">
          <h3>What will run</h3>
          <ul className="muted" style={{ lineHeight: 1.8, paddingLeft: 18 }}>
            <li><b>{cat.use_cases[w.use_case].name}</b>{w.use_case === "standard" ? ` · ${cat.prompt_sets[w.prompt_set].label}, ${w.answer_tokens}-token answers` : w.use_case === "translation" ? ` · ${cat.text_sizes[w.text_size]?.label ?? w.text_size}: ${w.directions.map((d) => cat.directions[d]?.label ?? d).join(", ")} · ${w.translation_items} items per direction in the quality pass` : ` · ${w.workloads.map((x) => cat.workloads[x].label).join("; ")} · ${w.test_items} documents in the accuracy check`}</li>
            <li>Model: <b>{w.models.map((id) => cat.models.find((m) => m.id === id)?.name).join(", ")}</b></li>
            <li>Hardware: {w.gpu && <Badge kind="gpu">{hwLabel(w.gpu.hardware, w.gpu.instance_type)}</Badge>} {w.cpus.map((c) => <Badge key={c.hardware} kind="cpu">{hwLabel(c.hardware, c.instance_type)}</Badge>)} · {w.parallel_targets ? "in parallel" : "one after another"} · {cat.purchase[w.purchase].label}{w.purchase === "spot" && w.spot_fallback ? `, On-Demand after ${w.spot_wait_minutes} min without Spot` : ""} · gives up on a hardware option after {w.launch_attempts} failed launch attempts</li>
            <li>Load: {loadSentence(w)}</li>
            <li>{depthSentence(w, cat)}</li>
            <li>Targets: {w.use_case === "dspm" ? cat.presets[w.preset].label : Object.entries(w.sla).filter(([, v]) => v).map(([k, v]) => `${{ p99_ttft_ms: "first word within", p50_tpot_ms: "each generated word within", p99_e2e_ms: w.use_case === "translation" ? "whole translation within" : cat.models.find((m) => w.models.includes(m.id) && (m.type === "decoder" || m.type === "encoder-decoder")) ? "whole answer within" : "whole decision within" }[k as string] ?? k} ${v} ms`).join(", ") || "none"}</li>
          </ul>
        </div>
        <div className="card soft">
          <h3>Estimate</h3>
          {!plan && !err && <Spinner label="Validating and estimating…" />}
          {plan?.estimate && (
            <div className="grid c3">
              <div className="metric"><div className="k">Wall-clock time</div><div className="v">{plan.estimate.wall_clock_hours.toFixed(1)} h</div><div className="d muted">assumed speeds until measured</div></div>
              <div className="metric"><div className="k">Cost</div><div className="v">{fmt.usd(plan.estimate.usd.total)}</div><div className="d muted">On-Demand rates; Spot is usually 30–70% less</div></div>
              <div className="metric"><div className="k">Grade</div><div className="v" style={{ fontSize: 18 }}>{plan.grade === "full" ? "Decision-grade" : "Quick check"}</div><div className="d muted">{plan.grade === "full" ? "200+ requests per measurement, repeated 3 times" : "indicative only"}</div></div>
            </div>
          )}
          {plan && (
            <table className="tbl mt">
              <thead><tr><th>Model</th><th>Hardware</th><th>Format</th><th className="num">Measurements</th></tr></thead>
              <tbody>{(plan.series ?? []).map((s, i) => <tr key={i}><td>{s.model}</td><td><Badge kind={s.accelerator}>{s.accelerator.toUpperCase()}</Badge> {s.instance_type} · {s.engine}</td><td>{s.precision}</td><td className="num">{s.measurements}</td></tr>)}</tbody>
            </table>
          )}
        </div>
      </div>
      {Array.from(new Map((plan?.pairing ?? []).map((p) => [p.reason, p])).values()).map((p, i) => <Note key={i} kind={p.fair ? "ok" : "warn"}><b>{p.fair ? "Fair comparison" : "Not a fair comparison"}</b> — {p.reason}.</Note>)}
      {(plan?.warnings ?? []).map((x, i) => <Note key={"w" + i} kind="warn">{x}</Note>)}
      {(plan?.problems ?? []).map((x, i) => <Note key={"p" + i} kind="bad">{x}</Note>)}
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
