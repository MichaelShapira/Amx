import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import Markdown from "react-markdown";
import { api, fmt, type CompareOut, type ExecSummary, type Group, type RunRow, type SummaryInfo } from "../api";
import { useCatalog } from "../App";
import { Badge, Help, Label, Metric, Note, Spinner, Toggle } from "../components/ui";
import { Curves, InstancesBar, Swatch, seriesKeys, seriesStyles } from "../components/Charts";

export default function Results() {
  const cat = useCatalog();
  const [params, setParams] = useSearchParams();
  const selected = (params.get("runs") ?? "").split(",").filter(Boolean);
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [infos, setInfos] = useState<Record<string, SummaryInfo>>({});
  const [out, setOut] = useState<CompareOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sla, setSla] = useState<Record<string, number | string | null> | null>(null);
  const [limit, setLimit] = useState(4);
  const [models, setModels] = useState<string[] | null>(null);
  const [view, setView] = useState<"all" | "one">("all");
  const [series, setSeries] = useState<string | null>(null);

  useEffect(() => { api.runs().then((r) => setRuns(r.filter((x) => x.status === "complete"))).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    selected.forEach((id) => { if (!infos[id]) api.summary(id).then((s) => setInfos((p) => ({ ...p, [id]: s }))).catch(() => undefined); });
  }, [selected.join(",")]);  // eslint-disable-line react-hooks/exhaustive-deps

  const key = useMemo(() => JSON.stringify({ selected, sla, limit, models }), [selected, sla, limit, models]);
  useEffect(() => {
    if (!selected.length) { setOut(null); return; }
    setErr(null);
    api.compare({ run_ids: selected, sla: sla ?? undefined, limit, models: models ?? undefined })
      .then((o) => { setOut(o); if (!sla) setSla(o.sla); })
      .catch((e) => setErr(e.message));
  }, [key]);  // eslint-disable-line react-hooks/exhaustive-deps

  const toggleRun = (id: string) => {
    const next = selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id];
    setParams(next.length ? { runs: next.join(",") } : {});
    setSla(null); setModels(null); setSeries(null);
  };
  const allModels = Array.from(new Set(selected.flatMap((id) => infos[id]?.models ?? [])));
  const useCase = infos[selected[0]]?.shape.use_case;
  const name = (mid: string) => cat?.models.find((m) => m.id === mid)?.name ?? mid;

  return (
    <div>
      <div className="topbar"><div><h1>Results</h1><div className="muted">Pick one run, or several to compare. Every selection is re-checked for comparability.</div></div></div>
      {err && <Note kind="bad">{err}</Note>}
      <div className="card">
        <h3>Runs <Help text="Finished runs with a report. Type part of a run's name, model or date to find it; pick one to see its results, more to compare them. Runs can only be compared when they answer the same question: same use case, same test data, same request shape." /></h3>
        {!runs && <Spinner />}
        {runs && runs.length === 0 && <div className="muted">No finished runs yet.</div>}
        {runs && runs.length > 0 && <RunPicker runs={runs} selected={selected} onToggle={toggleRun} onClear={() => { setParams({}); setSla(null); setModels(null); setSeries(null); }} />}
        {selected.length > 0 && (
          <table className="tbl mt">
            <thead><tr><th>Run</th><th>Use case</th><th>Data</th><th>Request shape</th><th>Concurrency</th><th>Grade</th><th>Models</th><th>Instances</th></tr></thead>
            <tbody>
              {selected.map((id) => { const s = infos[id]; if (!s) return <tr key={id}><td colSpan={8}><Spinner /></td></tr>; const sh = s.shape; return (
                <tr key={id}><td><b>{id}</b></td><td>{cat?.use_cases[sh.use_case]?.name}</td>
                  <td className="small">{sh.use_case === "standard" ? `${cat?.prompt_sets[sh.prompt_set!]?.label}, ${sh.answer_tokens}-token answers` : `${(sh.workloads ?? []).map((w) => cat?.workloads[w]?.label ?? w).join("; ")} · dataset ${sh.dataset}`}</td>
                  <td>{sh.request_mode}</td><td>{sh.concurrency.join(", ")}</td>
                  <td><Badge kind={sh.grade === "full" ? "ok" : "warn"}>{sh.grade === "full" ? "decision-grade" : "quick check"}</Badge></td>
                  <td className="small">{s.models.map(name).join(", ")}</td><td className="small">{s.instance_types.join(", ")}</td></tr>); })}
            </tbody>
          </table>
        )}
      </div>

      {out && (
        <>
          <div className="card mt">
            <h3>Can these be compared? <Help text="Re-evaluated on every selection. Blockers mean the runs answer different questions (different use case, data, answer length or request shape) and the tool refuses to put their numbers side by side. Warnings are differences to keep in mind. The fairness lines check that every GPU and CPU instance pair sits within half to double of each other's hourly price." /></h3>
            {selected.length < 2 && <Note kind="ok">One run selected: all its measurements come from the same test, so they are comparable by construction.</Note>}
            {selected.length >= 2 && out.comparability.ok && <Note kind="ok"><b>Comparable.</b> Same use case, same test data and request shape across the {selected.length} runs.</Note>}
            {out.comparability.blockers.map((b, i) => <Note key={i} kind="bad"><b>Not comparable.</b> {b}</Note>)}
            {out.analysis?.not_run.map((n, i) => {
              const sideMeasured = out.analysis!.groups.some((g) => g.points.some((p) => p.side === n.side.toLowerCase()));
              return (
                <Note key={"nr" + i} kind="bad">
                  <b>The {n.side} side did not run{selected.length > 1 ? ` in ${n.run_id}` : ""}.</b> {n.instance_type ? `${n.instance_type}: ` : ""}{n.reason}
                  {!sideMeasured && <> <b>What follows is a one-sided picture, not a GPU-vs-CPU comparison.</b> Start the run again later, pick another {n.side} size, or raise the number of launch attempts before giving up.</>}
                </Note>
              );
            })}
            {out.comparability.warnings.map((b, i) => <Note key={"w" + i} kind="warn">{b}</Note>)}
            {out.analysis?.pairing.map((p, i) => <Note key={"p" + i} kind={p.fair ? "ok" : "warn"}><b>{p.fair ? "Fair instance pairing" : "Unfair instance pairing"}</b> — {p.reason}.</Note>)}
            {out.analysis?.smoke && <Note kind="warn">Quick check: these numbers rest on fewer than 200 requests per measurement, or the test was not repeated 3 times. They show the shape of the result, not a basis for a decision.</Note>}
          </div>

          {out.analysis && (
            <>
              <div className="card mt">
                <div className="row between">
                  <h3 style={{ margin: 0 }}>Your targets <Help text="Everything below is judged against these. Pre-filled with the targets the run was started with; change them to see how the verdict moves." /></h3>
                  <div className="row small">
                    <Label help="The maximum number of CPU instances per GPU you would accept. Within it, the CPU side is a viable alternative; above it, conditional.">Max CPU instances per GPU</Label>
                    <input type="number" min={1} max={64} value={limit} onChange={(e) => setLimit(Number(e.target.value) || 4)} style={{ width: 70 }} />
                  </div>
                </div>
                {useCase === "dspm" ? (
                  <div className="row mt">
                    {cat && Object.entries(cat.presets).map(([k, p]) => (
                      <button key={k} className={"chip " + (sla?.preset === k ? "on" : "")} onClick={() => setSla({ p99_e2e_ms: p.p99_e2e_ms, max_error_pct: 1, preset: k })}>{p.label} · ≤ {fmt.n(p.p99_e2e_ms)} ms</button>
                    ))}
                  </div>
                ) : out.analysis && out.analysis.groups.length > 0 && out.analysis.groups.every((g) => g.units.lat_sla === "p99_e2e_ms") ? (
                  <div className="grid c3 mt">
                    <div><Label help="A classifier gives its whole answer at once, so this is the one time limit that applies: 99 of 100 decisions within this many milliseconds. Empty = no limit.">Whole decision within (ms)</Label><input type="number" value={(sla?.p99_e2e_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_ttft_ms: null, p50_tpot_ms: null, p99_e2e_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div className="small muted" style={{ gridColumn: "span 2", alignSelf: "end" }}>These models generate no words, so the first-word and per-word targets do not apply.</div>
                  </div>
                ) : (
                  <div className="grid c3 mt">
                    <div><Label help="Longest acceptable wait before the answer starts, 99 of 100 requests. Empty = no limit.">First word within (ms)</Label><input type="number" value={(sla?.p99_ttft_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_ttft_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div><Label help="Median time between generated tokens. Empty = no limit.">Time per generated word (ms)</Label><input type="number" value={(sla?.p50_tpot_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p50_tpot_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div><Label help="Optional overall limit per request, 99th percentile. Empty = no limit.">Whole answer within (ms)</Label><input type="number" value={(sla?.p99_e2e_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_e2e_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                  </div>
                )}
                {allModels.length > 1 && (
                  <div className="mt">
                    <Label help="Show only some of the models in the selected runs.">Models</Label>
                    <div className="chips">{allModels.map((m) => <button key={m} className={"chip " + ((models ?? allModels).includes(m) ? "on" : "")} onClick={() => { const cur = models ?? allModels; setModels(cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]); }}>{name(m)}</button>)}</div>
                  </div>
                )}
                <div className="row mt">
                  <Label help="All: every hardware option and number format on the same charts. One: a single series at a time, for reading one machine's numbers.">View</Label>
                  <Toggle value={view} onChange={(v) => setView(v)} options={[{ value: "all", label: "All hardware options" }, { value: "one", label: "One at a time" }]} />
                </div>
              </div>
              <ExecutiveSummary runIds={selected} sla={sla} limit={limit} models={models} />
              {out.analysis.groups.map((g) => <GroupView key={g.key} g={g} view={view} series={series} setSeries={setSeries} limit={limit} target={(sla?.[g.units.lat_sla] as number | null | undefined) ?? null} />)}
              {out.analysis.warnings.length > 0 && (
                <div className="card mt"><h3>Notes from the reports</h3><ul className="muted small">{Array.from(new Set(out.analysis.warnings)).map((w, i) => <li key={i}>{w}</li>)}</ul></div>
              )}
              <div className="row mt">{selected.map((id) => <button key={id} type="button" className="btn sm" onClick={() => api.downloadReport(id).catch((e) => setErr((e as Error).message))}>Download report: {id}</button>)}</div>
            </>
          )}
        </>
      )}
    </div>
  );
}

function SideCard({ kind, c, other, g }: { kind: "gpu" | "cpu"; c: Group["best_gpu"]; other: Group["best_gpu"]; g: Group }) {
  const cat = useCatalog();
  const missing = g.missing?.[kind];
  if (!c && missing) return <div className={`side-card ${kind} missing`}><h3><Badge kind="bad">{kind.toUpperCase()} did not run</Badge></h3><div className="small">{missing}</div><div className="muted small mt">Nothing on this side was measured, so the {kind === "gpu" ? "CPU" : "GPU"} card next to it has no comparison figures.</div></div>;
  if (!c) return <div className={`side-card ${kind}`}><h3><Badge kind={kind}>{kind.toUpperCase()}</Badge> No measurement meets your targets</h3><div className="muted small">Relax the targets or look at the charts below to see how far off it is.</div></div>;
  const r = (a: number | null, b: number | null | undefined) => (a && b ? a / b : null);
  const isCpu = kind === "cpu";
  const thrR = r(c.throughput, other?.throughput), latR = r(c.latency, other?.latency), costR = r(c.cost, other?.cost);
  const eng = cat?.engines[c.engine]?.label ?? c.engine;
  return (
    <div className={`side-card ${kind}`}>
      <h3><Badge kind={kind}>{kind.toUpperCase()}</Badge> {c.instance_type} · {eng} · {cat?.precisions[c.precision]?.label ?? c.precision}</h3>
      <div className="small muted">at {c.conc} concurrent request{c.conc === 1 ? "" : "s"}{c.batch > 1 ? `, batches of ${c.batch}` : ""} · {fmt.usd(c.usd_per_hour)}/hour{c.amx ? ` · AMX ${c.amx}` : ""}{c.run_id ? ` · run ${c.run_id}` : ""}</div>
      <div className="grid c3 mt">
        <Metric k="Throughput" v={`${fmt.n(c.throughput)} ${g.units.thr_label}`} d={isCpu && thrR != null ? `${fmt.x(thrR, 2)} of GPU` : undefined} dGood={isCpu && thrR != null ? thrR >= 1 : null} help={`${g.units.thr_label} across all concurrent requests. Higher is better; this is the capacity figure.`} />
        <Metric k="Responsiveness" v={`${fmt.n(c.latency)} ms`} d={isCpu && latR != null ? `${fmt.x(latR)} GPU's wait` : undefined} dGood={isCpu && latR != null ? latR <= 1 : null} help={`${g.units.lat_label}. Lower is better.`} />
        <Metric k="Cost per unit" v={fmt.usd(c.cost)} d={isCpu && costR != null ? `${fmt.x(costR)} GPU's cost` : undefined} dGood={isCpu && costR != null ? costR <= 1 : null} help={`${g.units.cost_label} at this load, On-Demand price. Lower is better.`} />
      </div>
      {c.accuracy && (
        <div className="small mt">Detection accuracy: recall <b>{fmt.pct(c.accuracy.recall)}</b> [{fmt.pct(c.accuracy.recall_ci[0])}–{fmt.pct(c.accuracy.recall_ci[1])}], false positives <b>{fmt.pct(c.accuracy.fpr)}</b>, F1 {c.accuracy.f1.toFixed(2)}{c.accuracy.agreement_with_gpu != null ? <> · agreement with GPU <b>{fmt.pct(c.accuracy.agreement_with_gpu)}</b></> : null}{!c.accuracy.fit && <Badge kind="warn">below the accuracy floors</Badge>}</div>
      )}
      {c.quality_delta != null && <div className="small mt">Answer quality loss vs GPU: <b>{(c.quality_delta * 100).toFixed(0)}%</b> <Help text="1 minus the similarity (ROUGE-L) of this option's answers to the GPU's answers on 50 fixed prompts. BF16 on CPU sets the noise floor; compressed formats are judged against it." /></div>}
    </div>
  );
}

function GroupView({ g, view, series, setSeries, limit, target }: { g: Group; view: "all" | "one"; series: string | null; setSeries: (s: string | null) => void; limit: number; target: number | null }) {
  const shown = view === "one" ? (series && g.series.includes(series) ? series : g.series[0]) : null;
  const pts = shown ? g.points.filter((p) => p.series === shown) : g.points;
  const styles = seriesStyles(seriesKeys(g.points));
  return (
    <div className="card mt">
      <h2>{g.label}</h2>
      {g.verdict && <Note kind={g.verdict.within_limit === true ? "ok" : g.verdict.within_limit === false ? "warn" : undefined}><b>Verdict.</b> {g.verdict.text}</Note>}
      <div className="grid c2 mt">
        <SideCard kind="gpu" c={g.best_gpu} other={g.best_cpu} g={g} />
        <SideCard kind="cpu" c={g.best_cpu} other={g.best_gpu} g={g} />
      </div>
      {g.equivalence.length > 0 && (
        <div className="mt">
          <h3>How many CPU instances replace one GPU? <Help text={`For every CPU measurement that meets your targets: the number of such instances needed to match the best GPU measurement's throughput, and the hourly cost of that many relative to the GPU. The dashed line is your limit of ${limit}.`} /></h3>
          <InstancesBar rows={g.equivalence} limit={limit} />
        </div>
      )}
      <div className="row between mt">
        <h3 style={{ margin: 0 }}>As the number of requests at a time grows</h3>
        {view === "one" && (
          <select value={shown ?? ""} onChange={(e) => setSeries(e.target.value)}>{g.series.map((s) => <option key={s} value={s}>{s}</option>)}</select>
        )}
      </div>
      <Curves points={pts} units={g.units} target={target} />
      <details className="mt"><summary className="small">All measurements in this group</summary>
        <table className="tbl mt">
          <thead><tr><th>Hardware · software · format</th><th className="num">Concurrent</th><th className="num">Batch</th><th className="num">{g.units.thr_label}</th><th className="num">{g.units.lat_label}</th><th className="num">median (ms)</th><th className="num">{g.units.cost_label}</th><th className="num">errors %</th><th>Meets targets</th><th>Run</th></tr></thead>
          <tbody>{g.points.map((p, i) => <tr key={i}><td><Swatch style={styles[p.series + (p.batch > 1 ? ` ·b${p.batch}` : "")]} /><Badge kind={p.side}>{p.side.toUpperCase()}</Badge> {p.series.replace(/^(GPU|CPU) · /, "")}</td><td className="num">{p.conc}</td><td className="num">{p.batch > 1 ? p.batch : "–"}</td><td className="num">{fmt.n(p.throughput)}</td><td className="num">{fmt.n(p.latency)}</td><td className="num">{fmt.n(p.e2e_p50)}</td><td className="num">{fmt.usd(p.cost)}</td><td className="num">{fmt.n(p.error_pct, 1)}</td><td>{p.meets ? <Badge kind="ok">yes</Badge> : <Badge kind="bad">{p.reason || "no"}</Badge>}</td><td className="tiny">{p.run_id}</td></tr>)}</tbody>
        </table>
      </details>
    </div>
  );
}


function ExecutiveSummary({ runIds, sla, limit, models }: { runIds: string[]; sla: Record<string, number | string | null> | null; limit: number; models: string[] | null }) {
  const [model, setModel] = useState<{ name: string; call_id: string } | null>(null);
  const [res, setRes] = useState<ExecSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.execModel().then(setModel).catch((e) => setErr(e.message)); }, []);
  useEffect(() => { setRes(null); }, [runIds.join(","), JSON.stringify(sla), limit, JSON.stringify(models)]);  // eslint-disable-line react-hooks/exhaustive-deps
  const go = async (force = false) => {
    setBusy(true); setErr(null);
    try { setRes(await api.execSummary({ run_ids: runIds, sla: sla ?? undefined, limit, models: models ?? undefined, force })); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <div className="card mt">
      <div className="row between">
        <div>
          <h3 style={{ margin: 0 }}>Executive summary <Help text="A written summary of the findings below, produced by the newest Claude Opus model available on Amazon Bedrock in this account. The model only sees the computed findings (verdicts, best measurements, instance counts, accuracy, caveats) and is instructed to use no number that is not in them. It is regenerated when you change the runs or the targets; each generation is one Bedrock call billed to this account." /></h3>
          <div className="small muted">{model ? `Written by ${model.name} (Bedrock, ${model.call_id})` : err ? "" : "Looking up the newest Opus on Bedrock…"}{res ? ` · ${res.generated_at.replace("T", " ").slice(0, 16)} UTC · ${res.seconds}s · ${res.usage.inputTokens + res.usage.outputTokens} tokens` : ""}</div>
        </div>
        <div className="row">
          {res && <button className="btn sm" onClick={() => go(true)} disabled={busy}>Rewrite</button>}
          {!res && <button className="btn primary" onClick={() => go(false)} disabled={busy || !model}>{busy ? <Spinner label="Writing…" /> : "Write the executive summary"}</button>}
        </div>
      </div>
      {err && <Note kind="bad">{err}</Note>}
      {res && <div className="prose mt"><Markdown>{res.markdown}</Markdown></div>}
    </div>
  );
}


/** Search-as-you-type run picker: selected runs as removable tags, a text box that filters by
 *  name, id, model or date, and the newest finished runs as suggestions when the box is empty. */
function RunPicker({ runs, selected, onToggle, onClear }: { runs: RunRow[]; selected: string[]; onToggle: (id: string) => void; onClear: () => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const byId = useMemo(() => Object.fromEntries(runs.map((r) => [r.run_id, r])), [runs]);
  const label = (r: RunRow) => (r.name && r.name !== r.run_id ? r.name : r.run_id);
  const matches = useMemo(() => {
    const terms = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const hay = (r: RunRow) => `${r.run_id} ${r.name ?? ""} ${r.model ?? ""} ${r.updated_at ?? ""} ${r.grade === "full" ? "decision-grade full" : "quick"}`.toLowerCase();
    const list = terms.length ? runs.filter((r) => terms.every((t) => hay(r).includes(t))) : runs;
    return list.filter((r) => !selected.includes(r.run_id)).slice(0, 12);
  }, [q, runs, selected]);
  useEffect(() => { setCursor(0); }, [q, open]);
  useEffect(() => {
    const onDoc = (e: MouseEvent) => { if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);
  const pick = (id: string) => { onToggle(id); setQ(""); setOpen(true); };
  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); setCursor((c) => Math.min(matches.length - 1, c + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(0, c - 1)); }
    else if (e.key === "Enter" && open && matches[cursor]) { e.preventDefault(); pick(matches[cursor].run_id); }
    else if (e.key === "Escape") { setOpen(false); }
    else if (e.key === "Backspace" && q === "" && selected.length) { onToggle(selected[selected.length - 1]); }
  };
  return (
    <div className="picker" ref={boxRef}>
      <div className="picker-box" onClick={() => setOpen(true)}>
        {selected.map((id) => { const r = byId[id]; return (
          <span key={id} className="tag">
            <b>{r ? label(r) : id}</b>{r?.model ? <span className="muted"> · {r.model}</span> : null}{r && r.grade !== "full" ? <Badge kind="warn">quick</Badge> : null}
            <button type="button" className="tag-x" aria-label={`Remove ${id}`} onClick={(e) => { e.stopPropagation(); onToggle(id); }}>×</button>
          </span>); })}
        <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); }} onFocus={() => setOpen(true)} onKeyDown={onKey}
          placeholder={selected.length ? "Add another run to compare…" : `Search ${runs.length} finished runs by name, model or date…`} aria-label="Search runs" />
        {selected.length > 0 && <button type="button" className="btn sm" onClick={(e) => { e.stopPropagation(); onClear(); }}>Clear</button>}
      </div>
      {open && (
        <div className="picker-list" role="listbox">
          {matches.length === 0 && <div className="muted small" style={{ padding: "8px 10px" }}>{q ? "No finished run matches." : "Every finished run is already selected."}</div>}
          {!q && matches.length > 0 && <div className="tiny muted" style={{ padding: "6px 10px 2px" }}>Most recent</div>}
          {matches.map((r, i) => (
            <div key={r.run_id} role="option" aria-selected={i === cursor} className={"picker-item " + (i === cursor ? "active" : "")} onMouseEnter={() => setCursor(i)} onMouseDown={(e) => { e.preventDefault(); pick(r.run_id); }}>
              <div><b>{label(r)}</b>{r.name && r.name !== r.run_id ? <span className="tiny muted"> {r.run_id}</span> : null}</div>
              <div className="small muted">{r.model ?? "–"} · {r.updated_at?.slice(0, 16).replace("T", " ") ?? ""} UTC · {r.grade === "full" ? "decision-grade" : "quick check"}{r.not_run?.length ? " · one side did not run" : ""}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
