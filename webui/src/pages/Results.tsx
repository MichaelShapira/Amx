import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import Markdown from "react-markdown";
import { type CompareOut, type CpuCaseCost, type DemandOption, type DemandPlan, type ExecSummary, type Group, type IntelReference, type Pricing, type RunRow, type RunUtilization, type Sizing, type SummaryInfo, type TypicalLoad, type WhereCpuWins, api, fmt } from "../api";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useCatalog } from "../App";
import { Badge, Help, Label, Metric, Note, Spinner, Toggle, instanceLine } from "../components/ui";
import { Curves, InstancesBar, Swatch, UtilizationCurves, UtilizationTimeline, seriesKeys, seriesStyles } from "../components/Charts";

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
  const [pricing, setPricing] = useState<Pricing>({ basis: "on_demand", cpu_usd_per_hour: null, gpu_usd_per_hour: null });
  const [demandRpm, setDemandRpm] = useState<number | null>(null);

  useEffect(() => { api.runs().then((r) => setRuns(r.filter((x) => x.status === "complete"))).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    selected.forEach((id) => { if (!infos[id]) api.summary(id).then((s) => setInfos((p) => ({ ...p, [id]: s }))).catch(() => undefined); });
  }, [selected.join(",")]);  // eslint-disable-line react-hooks/exhaustive-deps

  const key = useMemo(() => JSON.stringify({ selected, sla, limit, models, pricing, demandRpm }), [selected, sla, limit, models, pricing, demandRpm]);
  useEffect(() => {
    if (!selected.length) { setOut(null); return; }
    setErr(null);
    api.compare({ run_ids: selected, sla: sla ?? undefined, limit, models: models ?? undefined, pricing, demand_rpm: demandRpm })
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
                  <td className="small">{sh.use_case === "standard" ? `${cat?.prompt_sets[sh.prompt_set!]?.label}, ${sh.answer_tokens}-token answers` : sh.use_case === "translation" ? `${cat?.text_sizes?.[sh.prompt_set!]?.label ?? sh.prompt_set}: ${(sh.directions ?? []).map((d) => cat?.directions?.[d]?.label ?? d).join(", ")} · corpus ${sh.dataset}` : sh.use_case === "embedding" ? `${cat?.text_sizes?.[sh.prompt_set!]?.label ?? sh.prompt_set}, EN/DE/ES, embedded` : sh.use_case === "intel" ? `Intel's benchmark, ${cat?.intel_io?.[sh.io!]?.label ?? sh.io} tokens` : `${(sh.workloads ?? []).map((w) => cat?.workloads[w]?.label ?? w).join("; ")} · dataset ${sh.dataset}`}</td>
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
                ) : useCase === "embedding" ? (
                  <div className="grid c3 mt">
                    <div><Label help="An embedding request returns all its vectors at once: 99 of 100 requests within this many milliseconds. Empty = no limit.">Whole request within (ms)</Label><input type="number" value={(sla?.p99_e2e_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_ttft_ms: null, p50_tpot_ms: null, p99_e2e_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div className="small muted" style={{ gridColumn: "span 2", alignSelf: "end" }}>Embeddings generate no words, so the first-word and per-word targets do not apply.</div>
                  </div>
                ) : useCase === "intel" ? (
                  <div className="grid c3 mt">
                    <div><Label help="Intel's criterion: median time between generated tokens at most 100 ms.">Time per generated token (ms)</Label><input type="number" value={(sla?.p50_tpot_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p50_tpot_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div><Label help="Intel's criterion: first token within 3,000 ms (applied here to 99 of 100 requests).">First token within (ms)</Label><input type="number" value={(sla?.p99_ttft_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_ttft_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div className="small muted" style={{ alignSelf: "end" }}>Intel reports throughput at the highest concurrency that passes both.</div>
                  </div>
                ) : useCase === "translation" ? (
                  <div className="grid c3 mt">
                    <div><Label help="Longest acceptable time for the whole translation, 99 of 100 requests. Empty = no limit.">Whole translation within (ms)</Label><input type="number" value={(sla?.p99_e2e_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_e2e_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div><Label help="Longest acceptable wait before the translation starts appearing, 99 of 100 requests. Empty = no limit.">First word within (ms)</Label><input type="number" value={(sla?.p99_ttft_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p99_ttft_ms: e.target.value ? Number(e.target.value) : null })} /></div>
                    <div><Label help="Median time between generated tokens. Empty = no limit.">Time per generated word (ms)</Label><input type="number" value={(sla?.p50_tpot_ms as number) ?? ""} onChange={(e) => setSla({ ...sla, p50_tpot_ms: e.target.value ? Number(e.target.value) : null })} /></div>
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
              <LoadAndPrices pricing={pricing} setPricing={setPricing} demandRpm={demandRpm} setDemandRpm={setDemandRpm} typical={out.analysis.typical_loads ?? []} priceLabel={out.analysis.pricing?.label} snapshot={out.analysis.pricing?.snapshot ?? null} />
              {out.analysis.where_cpu_wins && !("error" in out.analysis.where_cpu_wins) && <WhereCpuWinsCard w={out.analysis.where_cpu_wins} />}
              {out.analysis.how_to_read && <div className="note mt"><b>How to read these results.</b> {out.analysis.how_to_read}</div>}
              <ExecutiveSummary runIds={selected} sla={sla} limit={limit} models={models} pricing={pricing} demandRpm={demandRpm} />
              {out.analysis.groups.map((g) => <GroupView key={g.key} g={g} view={view} series={series} setSeries={setSeries} limit={limit} target={(sla?.[g.units.lat_sla] as number | null | undefined) ?? null} />)}
              {out.analysis.intel_reference && out.analysis.intel_reference.length > 0 && <IntelReferenceTable rows={out.analysis.intel_reference} groups={out.analysis.groups} />}
              <RunUtilization runIds={selected} />
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

const COST_KIND: Record<CpuCaseCost["kind"], { badge: "ok" | "warn" | "bad" | undefined; label: string }> = {
  cpu_everywhere: { badge: "ok", label: "CPU cheaper at every load" }, cpu_below_crossover: { badge: "ok", label: "CPU cheaper at low to moderate load" },
  gpu_always: { badge: "bad", label: "GPU cheaper at every load" }, gpu_only: { badge: "bad", label: "only the GPU met the targets" },
  cpu_only: { badge: "warn", label: "only the CPU met the targets" }, none: { badge: undefined, label: "nothing to compare" },
};

/** The section the reader came for: exactly where the CPU is the better choice, computed from the measurements, and where it is not. */
function WhereCpuWinsCard({ w }: { w: WhereCpuWins }) {
  const money = (m: number) => `$${Math.round(m).toLocaleString()}`;
  return (
    <div className="card mt" style={{ borderColor: w.any_benefit ? "var(--ok)" : "var(--bad)", borderWidth: 2 }}>
      <h2 style={{ margin: 0 }}>Where the CPU is the better choice <Help text="Computed from the selected runs, never asserted. Cost: from each question's measurements, the load range below which the cheapest CPU option that meets your targets costs less than the cheapest GPU option, with worked examples at typical loads (at the prices chosen above). Availability: the runs' own capacity log, how often each side's launch was refused. Smallest unit: the cheapest instance that held the model and met the targets on each side. Everything the CPU does not win is listed too." /></h2>
      <Note kind={w.any_benefit ? "ok" : "bad"}><b>{w.headline}</b></Note>
      <h3 className="mt">1 · Cost at your kind of load</h3>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {w.cost.map((c) => {
          const range = c.crossover_rpm && (c.kind === "cpu_below_crossover") ? `CPU cheaper below ~${Math.round(c.crossover_rpm).toLocaleString()} requests/minute; GPU cheaper above` : COST_KIND[c.kind].label;
          return (
            <div key={c.key} className="tile" style={{ cursor: "default" }}>
              <div className="row between" style={{ alignItems: "baseline", flexWrap: "wrap", gap: 8 }}>
                <div className="t" style={{ margin: 0 }}>{c.group}</div>
                <Badge kind={COST_KIND[c.kind].badge}>{range}</Badge>
              </div>
              {c.examples.length > 0 ? (
                <table className="tbl mt" style={{ fontSize: 13 }}>
                  <thead><tr><th>Load</th><th>Cheapest CPU option</th><th className="num">per month</th><th>Cheapest GPU option</th><th className="num">per month</th><th className="num">CPU saves</th></tr></thead>
                  <tbody>
                    {c.examples.map((e) => (
                      <tr key={e.label}>
                        <td>{e.label.replace(/:.*$/, "")} <span className="muted">· {e.rpm.toLocaleString()} req/min</span></td>
                        <td><b>{e.cpu}</b>{e.estimated && <> <Badge kind="warn">estimated</Badge></>}</td>
                        <td className="num">{money(e.cpu_usd_per_month)}</td>
                        <td>{e.gpu}</td>
                        <td className="num">{money(e.gpu_usd_per_month)}</td>
                        <td className="num"><b>{e.ratio ? `${e.ratio.toFixed(1)}x` : "–"}</b></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="small mt">{c.text}</div>
              )}
              {c.examples.length > 0 && c.examples.some((e) => e.estimated) && (
                <div className="tiny muted mt">"estimated" = a smaller size of the measured CPU family, sized by core count and checked for memory, not yet measured; one run at that size confirms it.</div>
              )}
            </div>
          );
        })}
      </div>
      <h3 className="mt">2 · Availability <Badge kind={w.availability.kind.startsWith("gpu_scarcer") ? "ok" : undefined}>{w.availability.kind === "gpu_scarcer" ? "real benefit in these runs" : w.availability.kind === "gpu_scarcer_spot" ? "mild edge (Spot only)" : w.availability.kind === "none" ? "no evidence" : "no advantage"}</Badge></h3>
      <div className="small">{w.availability.text}</div>
      {w.granularity && (<><h3 className="mt">3 · Smallest unit you can buy <Badge kind="ok">{w.granularity.ratio.toFixed(0)}x smaller</Badge></h3><div className="small">{w.granularity.text}</div></>)}
      <h3 className="mt">{w.granularity ? "4" : "3"} · Not measured here, but real</h3>
      <ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>{w.structural.map((x) => <li key={x.title}><b>{x.title}.</b> {x.text}</li>)}</ul>
      <h3 className="mt">Where the CPU is not the better choice</h3>
      <div className="small">{w.not_won.text}</div>
      {w.not_won.peak_throughput.length > 0 && <ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>{w.not_won.peak_throughput.map((t) => <li key={t}>{t}</li>)}</ul>}
      {w.losses.length > 0 && <ul className="small" style={{ margin: "4px 0 0", paddingLeft: 18 }}>{w.losses.map((l) => <li key={l.key}><b>{l.group}:</b> {l.text}</li>)}</ul>}
    </div>
  );
}

const BASES: { value: string; label: string; help: string }[] = [
  { value: "on_demand", label: "On-Demand (pay as you go)", help: "The public hourly price with no commitment. The reports' own numbers use it." },
  { value: "compute_1y", label: "Compute Savings Plan · 1 year", help: "Commit to an hourly spend for one year, no upfront payment; applies to any instance family and region. Real rates from AWS, refreshed with the instance catalog." },
  { value: "compute_3y", label: "Compute Savings Plan · 3 years", help: "The same, for three years: the deepest discount that still lets you change instance family." },
  { value: "ec2_instance_1y", label: "EC2 Instance Savings Plan · 1 year", help: "Commit to one instance family in one region for a year, no upfront: cheaper than the Compute plan, less flexible." },
  { value: "ec2_instance_3y", label: "EC2 Instance Savings Plan · 3 years", help: "The deepest discount AWS publishes for these families." },
];

/** The two inputs that turn "how many CPUs match one GPU" into a decision: the load to serve and the prices to use. */
function LoadAndPrices({ pricing, setPricing, demandRpm, setDemandRpm, typical, priceLabel, snapshot }: {
  pricing: Pricing; setPricing: (p: Pricing) => void; demandRpm: number | null; setDemandRpm: (n: number | null) => void;
  typical: TypicalLoad[]; priceLabel?: string; snapshot: string | null;
}) {
  const [text, setText] = useState<string>(demandRpm != null ? String(demandRpm) : "");
  const apply = (v: string) => { setText(v); const n = Number(v); setDemandRpm(v.trim() && n > 0 ? n : null); };
  return (
    <div className="card mt">
      <h3 style={{ margin: 0 }}>Your load and your prices <Help text="The capacity verdict above asks how many CPU instances match one GPU's PEAK throughput, which the GPU always wins on AWS pricing. The decision is the cheapest option that meets your targets at the load you actually have, at the prices you actually pay. Set both here; every group below gets an 'At your load' card and the executive summary uses them." /></h3>
      <div className="grid c2 mt">
        <div>
          <Label help="How many requests your application sends per minute at the level you want to size for (peak hour, not average). Pick a typical figure or type your own; the field always wins.">Load to serve (requests per minute)</Label>
          <div className="row">
            <input type="number" min={1} value={text} placeholder="e.g. 600" onChange={(e) => apply(e.target.value)} style={{ width: 140 }} />
            {typical.map((t) => <button key={t.key} className={"chip " + (demandRpm === t.requests_per_minute ? "on" : "")} title={t.help} onClick={() => apply(String(t.requests_per_minute))}>{t.label}</button>)}
            {demandRpm != null && <button className="btn sm" onClick={() => apply("")}>Clear</button>}
          </div>
          <div className="tiny muted mt">Empty: the verdicts assume you need a GPU's full peak. A request here is one API call, whatever it carries (one chat turn, one document, one batch of texts).</div>
        </div>
        <div>
          <Label help="Which hourly prices to use on both sides. Savings Plans are real published rates (no upfront, Linux, shared tenancy); they lower both sides, and the CPU side a little more. Your own rate overrides the basis for that side: enter 0 for capacity you already pay for.">Prices</Label>
          <select value={pricing.basis} onChange={(e) => setPricing({ ...pricing, basis: e.target.value })} title={BASES.find((b) => b.value === pricing.basis)?.help}>
            {BASES.map((b) => <option key={b.value} value={b.value}>{b.label}</option>)}
          </select>
          <div className="row mt small">
            <span>Your own rate, $/hour:</span>
            <label className="row" style={{ gap: 4 }}>CPU <input type="number" min={0} step={0.01} placeholder="basis" value={pricing.cpu_usd_per_hour ?? ""} onChange={(e) => setPricing({ ...pricing, cpu_usd_per_hour: e.target.value === "" ? null : Number(e.target.value) })} style={{ width: 90 }} /></label>
            <label className="row" style={{ gap: 4 }}>GPU <input type="number" min={0} step={0.01} placeholder="basis" value={pricing.gpu_usd_per_hour ?? ""} onChange={(e) => setPricing({ ...pricing, gpu_usd_per_hour: e.target.value === "" ? null : Number(e.target.value) })} style={{ width: 90 }} /></label>
          </div>
          <div className="tiny muted mt">Using: {priceLabel ?? "On-Demand"}{pricing.cpu_usd_per_hour != null ? ` · CPU at your $${pricing.cpu_usd_per_hour}/h` : ""}{pricing.gpu_usd_per_hour != null ? ` · GPU at your $${pricing.gpu_usd_per_hour}/h` : ""}{snapshot ? ` · AWS prices as of ${snapshot.slice(0, 10)}` : ""}.</div>
        </div>
      </div>
    </div>
  );
}

/** One group's cheapest-at-your-load result: the two sides sized for the demand, the crossover, and a cost-vs-load chart. */
function AtYourLoad({ plan, label }: { plan: DemandPlan; label: string }) {
  const cat = useCatalog();
  const money = (v: number) => `$${v.toFixed(2)}/h · $${Math.round(v * 730).toLocaleString()}/month`;
  const one = (o: DemandOption | null, title: string) => {
    if (!o) return <div className="tile" style={{ cursor: "default" }}><div className="t">{title}</div><div className="small muted">No option on this side met your targets.</div></div>;
    return (
      <div className={`tile ${o.side}`} style={{ cursor: "default" }}>
        <div className="t"><Badge kind={o.side}>{o.side.toUpperCase()}</Badge> {title}</div>
        <div><b>{o.instances} × {o.instance_type}</b> · {cat?.engines[o.engine]?.label ?? o.engine} · {cat?.precisions[o.precision]?.label ?? o.precision}{o.estimated && <> <Badge kind="warn">estimated size</Badge></>}</div>
        <div className="small mt"><b>{money(o.usd_per_hour)}</b> <span className="muted">({o.price_label})</span></div>
        <div className="tiny muted mt">Capacity {o.capacity_basis}. {o.headroom != null ? `At your load it would run at about ${Math.max(1, Math.round(100 / (1 + o.headroom)))}% of that capacity.` : ""}</div>
        {o.estimated && <div className="tiny mt" style={{ color: "var(--warn)" }}>Not measured: this size was derived from the measured one by core count and checked for memory. Run it (New benchmark, this size) to confirm before deciding.</div>}
      </div>
    );
  };
  const ratio = plan.cheapest_gpu && plan.cheapest_cpu ? (plan.cheapest_gpu.usd_per_hour / plan.cheapest_cpu.usd_per_hour) : null;
  return (
    <div className="mt">
      <h3>At your load: {Math.round(plan.demand_rpm).toLocaleString()} requests per minute <Help text="Each option's capacity is the highest request rate at which it still met your targets in the measurements; the number of instances is your load divided by that capacity, rounded up; the cost is that many instances at the chosen prices. Smaller CPU sizes than the one measured are estimated from core count and marked; run them to confirm." /></h3>
      <Note kind={plan.cheapest?.side === "cpu" ? "ok" : undefined}><b>{plan.cheapest?.side === "cpu" ? "The CPU side is the cheaper one here." : "The GPU side is the cheaper one here."}</b> {plan.text}{ratio != null && ratio > 1 ? ` The GPU option costs ${ratio.toFixed(1)}x the CPU option.` : ""}</Note>
      <div className="grid c2 mt">
        {one(plan.cheapest_cpu, "Cheapest CPU option")}
        {one(plan.cheapest_gpu, "Cheapest GPU option")}
      </div>
      {plan.crossover && <div className="small mt"><b>Crossover.</b> {plan.crossover.text} <Help text="Found by sizing both sides for every load from zero to beyond the GPU's measured capacity. Below the crossover a small CPU slice is cheaper than the smallest GPU instance; above it the GPU's throughput per dollar wins." /></div>}
      <CostCurve curve={plan.curve} demandRpm={plan.demand_rpm} crossover={plan.crossover?.rpm ?? null} label={label} />
    </div>
  );
}

function CostCurve({ curve, demandRpm, crossover, label }: { curve: DemandPlan["curve"]; demandRpm: number; crossover: number | null; label: string }) {
  const data = curve.map((c) => ({ rpm: Math.round(c.rpm), GPU: c.gpu_usd_per_hour, "CPU (best, incl. estimated sizes)": c.cpu_usd_per_hour, "CPU (measured sizes only)": Number.isFinite(c.cpu_measured_usd_per_hour ?? NaN) ? c.cpu_measured_usd_per_hour : null }));
  return (
    <div className="mt" style={{ height: 240 }}>
      <div className="small muted">Cheapest hourly cost per side as the load grows ({label}). Dashed vertical line: your load{crossover ? "; dotted: the crossover" : ""}.</div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          <XAxis dataKey="rpm" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toLocaleString()} label={{ value: "requests per minute", position: "insideBottom", offset: -2, fontSize: 11 }} tick={{ fontSize: 11 }} />
          <YAxis tickFormatter={(v) => `$${v}`} tick={{ fontSize: 11 }} width={56} />
          <Tooltip formatter={(v: any) => (v == null ? "–" : `$${Number(v).toFixed(2)}/h`)} labelFormatter={(v) => `${Number(v).toLocaleString()} requests/minute`} />
          <Legend />
          <ReferenceLine x={Math.round(demandRpm)} stroke="var(--fg)" strokeDasharray="6 4" />
          {crossover != null && <ReferenceLine x={Math.round(crossover)} stroke="var(--fg-3)" strokeDasharray="2 4" />}
          <Line type="stepAfter" dataKey="GPU" stroke="var(--gpu)" dot={false} strokeWidth={2} isAnimationActive={false} />
          <Line type="stepAfter" dataKey="CPU (best, incl. estimated sizes)" stroke="var(--cpu)" dot={false} strokeWidth={2} isAnimationActive={false} />
          <Line type="stepAfter" dataKey="CPU (measured sizes only)" stroke="var(--cpu)" dot={false} strokeWidth={1} strokeDasharray="4 3" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
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
      <div className="small muted">at {c.conc} concurrent request{c.conc === 1 ? "" : "s"}{c.batch > 1 ? `, batches of ${c.batch}` : ""} · {fmt.usd(c.usd_per_hour)}/hour{c.price_label && c.price_label !== "On-Demand" ? ` (${c.price_label})` : ""}{c.amx ? ` · AMX ${c.amx}` : ""}{c.run_id ? ` · run ${c.run_id}` : ""}</div>
      {c.savings && c.savings.length > 1 && (
        <div className="tiny muted mt">With a commitment: {c.savings.filter((x) => x.basis !== "on_demand").map((x) => `${x.label.replace(", no upfront", "")} $${x.usd_per_hour.toFixed(2)}/h (−${x.saving_pct}%)`).join(" · ")} <Help text="Published AWS Savings Plan rates for this instance type (no upfront payment, Linux, shared tenancy), next to the On-Demand price the measurement used. Choose a basis under 'Your load and your prices' to apply one to every number on this page." /></div>
      )}
      <div className="grid c3 mt">
        <Metric k="Throughput" v={`${fmt.n(c.throughput)} ${g.units.thr_label}`} d={isCpu && thrR != null ? `${fmt.x(thrR, 2)} of GPU` : undefined} dGood={isCpu && thrR != null ? thrR >= 1 : null} help={`${g.units.thr_label} across all concurrent requests. Higher is better; this is the capacity figure.`} />
        <Metric k="Responsiveness" v={`${fmt.n(c.latency)} ms`} d={isCpu && latR != null ? `${fmt.x(latR)} GPU's wait` : undefined} dGood={isCpu && latR != null ? latR <= 1 : null} help={`${g.units.lat_label}. Lower is better.`} />
        <Metric k="Cost per unit" v={fmt.usd(c.cost)} d={isCpu && costR != null ? `${fmt.x(costR)} GPU's cost` : undefined} dGood={isCpu && costR != null ? costR <= 1 : null} help={`${g.units.cost_label} at this load, at the chosen prices (${c.price_label ?? "On-Demand"}). Lower is better.`} />
      </div>
      {c.accuracy && (
        <div className="small mt">Detection accuracy: recall <b>{fmt.pct(c.accuracy.recall)}</b> [{fmt.pct(c.accuracy.recall_ci[0])}–{fmt.pct(c.accuracy.recall_ci[1])}], false positives <b>{fmt.pct(c.accuracy.fpr)}</b>, F1 {c.accuracy.f1.toFixed(2)}{c.accuracy.agreement_with_gpu != null ? <> · agreement with GPU <b>{fmt.pct(c.accuracy.agreement_with_gpu)}</b></> : null}{!c.accuracy.fit && <Badge kind="warn">below the accuracy floors</Badge>}</div>
      )}
      {c.translation && (
        <div className="small mt">Translation quality: chrF <b>{fmt.n(c.translation.chrf)}</b> vs reference <Help text="chrF: character-level similarity (0-100) of this option's translations to the reference text, averaged over the items of the quality pass. A paraphrase lowers it for every option alike, so read it relative to the other options, not as an absolute grade." />{c.translation.chrf_vs_gpu != null ? <> · <b>{fmt.n(c.translation.chrf_vs_gpu)}</b> vs the GPU's translations{c.translation.exact_match_vs_gpu != null ? ` (${fmt.pct(c.translation.exact_match_vs_gpu)} identical)` : ""}</> : null}{c.translation.length_ratio != null ? <> · length {fmt.x(c.translation.length_ratio, 2)} of reference</> : null}{(c.translation.truncated_pct ?? 0) > 0 ? <> · <Badge kind="warn">{fmt.n(c.translation.truncated_pct)}% cut at the token limit</Badge></> : null}{c.translation.flag && <Badge kind="bad">quality loss: {c.translation.flag}</Badge>}</div>
      )}
      {c.embedding && (
        <div className="small mt">Embedding quality: {c.embedding.cosine_mean != null ? <>cosine to the GPU's vectors <b>{c.embedding.cosine_mean.toFixed(4)}</b> (worst {c.embedding.cosine_min?.toFixed(4)}) <Help text="Cosine similarity between this option's vector and the GPU's vector for the same text; 1.0000 = identical direction. Differences come from BF16 arithmetic order, not from the model." /> · </> : null}retrieval DE→EN <b>{fmt.pct(c.embedding.recall_de)}</b>, ES→EN <b>{fmt.pct(c.embedding.recall_es)}</b> <Help text="For every German or Spanish text in the pool, is the nearest English text (by cosine) its own translation? A task score with a known answer; the same model on two machines should score alike." />{c.embedding.flag && <Badge kind="bad">quality loss: {c.embedding.flag}</Badge>}</div>
      )}
      {c.quality_delta != null && !c.translation && !c.embedding && <div className="small mt">Answer quality loss vs GPU: <b>{(c.quality_delta * 100).toFixed(0)}%</b> <Help text="1 minus the similarity (ROUGE-L) of this option's answers to the GPU's answers on 50 fixed prompts. BF16 on CPU sets the noise floor; compressed formats are judged against it." /></div>}
      {c.quality_flag && <div className="note warn small mt">Quality flag: {c.quality_flag}. <Help text="A warning, not an exclusion: this option stays in the rankings. The flag says its answers diverged from the reference more than the BF16 floor allows; for chat that measures different wording, not necessarily worse answers. Translation (chrF), embedding (cosine, recall@1) and detection (accuracy) flags measure the task itself." /></div>}
    </div>
  );
}

function IntelReferenceTable({ rows, groups }: { rows: IntelReference[]; groups: Group[] }) {
  const cat = useCatalog();
  const ours = groups.flatMap((g) => g.points.filter((p) => p.meets && p.throughput).map((p) => ({ ...p, label: g.label })));
  const cores = (t: string) => Math.max(1, Math.floor((cat?.instances[t]?.vcpus ?? 0) / 2));
  return (
    <div className="card mt">
      <h3>Intel's published figures for this model <Help text="Intel's own measurements from its Software Catalog (two-socket Xeon 6 servers, vLLM v0.18, tensor parallel 4), at the highest concurrency that kept TPOT at or below 100 ms and TTFT at or below 3 s. Per core is the honest comparison: an AWS slice has a fraction of a socket's cores and of its memory bandwidth, and Intel's concurrency is several times higher." /></h3>
      <table className="tbl mt">
        <thead><tr><th>Source</th><th>System</th><th className="num">Cores</th><th>Memory</th><th className="num">Concurrency</th><th className="num">tok/s</th><th className="num">tok/s per core</th><th className="num">TPOT ms</th><th className="num">TTFT ms</th></tr></thead>
        <tbody>
          {ours.filter((p) => p.side === "cpu").sort((a, b) => (b.throughput ?? 0) - (a.throughput ?? 0)).slice(0, 3).map((p, i) => (
            <tr key={"ours" + i}><td><Badge kind="cpu">this run</Badge></td><td>{p.instance_type} · {p.engine} · {p.precision}</td><td className="num">{cores(p.instance_type)}</td><td>{cat?.instances[p.instance_type]?.memory_gib} GiB</td><td className="num">{p.conc}</td><td className="num">{fmt.n(p.throughput)}</td><td className="num"><b>{(p.throughput! / cores(p.instance_type)).toFixed(1)}</b></td><td className="num">{fmt.n(p.tpot_p50)}</td><td className="num">{fmt.n(p.ttft_p99)} (p99)</td></tr>
          ))}
          {ours.filter((p) => p.side === "gpu").sort((a, b) => (b.throughput ?? 0) - (a.throughput ?? 0)).slice(0, 1).map((p, i) => (
            <tr key={"gpu" + i}><td><Badge kind="gpu">this run</Badge></td><td>{p.instance_type} · {p.engine} · {p.precision}</td><td className="num">—</td><td>{cat?.instances[p.instance_type]?.gpu_memory_gib} GiB GPU</td><td className="num">{p.conc}</td><td className="num">{fmt.n(p.throughput)}</td><td className="num">—</td><td className="num">{fmt.n(p.tpot_p50)}</td><td className="num">{fmt.n(p.ttft_p99)} (p99)</td></tr>
          ))}
          {rows.map((r, i) => (
            <tr key={i}><td><Badge>Intel</Badge> <span className="tiny">{r.tested}</span></td><td>{r.sockets}× {r.cpu_model}</td><td className="num">{r.total_cores}</td><td>{r.memory_gb} GB DDR5-{r.ddr5_mt_s}</td><td className="num">{r.max_concurrency}</td><td className="num">{fmt.n(r.throughput_tok_s)}</td><td className="num"><b>{r.tok_s_per_core?.toFixed(1)}</b></td><td className="num">{fmt.n(r.tpot_ms)}</td><td className="num">{fmt.n(r.ttft_ms)} (mean)</td></tr>
          ))}
        </tbody>
      </table>
      <div className="tiny muted mt">Intel's TTFT is a mean; this tool's is the 99th percentile, which is stricter. Intel's throughput is at the highest passing concurrency; this run's rows are the measured concurrencies that pass your targets.</div>
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
      {g.demand_plan && !("error" in g.demand_plan) && g.demand_plan.options.length > 0 && <AtYourLoad plan={g.demand_plan} label={g.label} />}
      {g.demand_plan && "error" in g.demand_plan && <Note kind="warn">The at-your-load plan could not be computed: {g.demand_plan.error}</Note>}
      {g.mechanics?.why && (
        <div className="note mt"><b>Why it came out this way.</b> {g.mechanics.why} <Help text="Computed from this run: tokens read and written per request, the share of time spent writing, the memory bandwidth each side reached (the model's weights stream once per written token) against what it has (STREAM measured on the CPU host; the vendor figure for the GPU), and the arithmetic rate at the heaviest load against each side's peak. Reading (prefill) is arithmetic, which AMX is built for; writing (decode) is memory bandwidth, where a price-matched CPU slice is furthest behind." /></div>
      )}
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
      <div className="hr" />
      <div className="row between">
        <h3 style={{ margin: 0 }}>Was the hardware the right size? <Help text="From the CloudWatch agent on each test host: how busy the GPU (GPU options) or the whole CPU (CPU options) was during each measurement, as the average of 10-second samples. The verdict reads the heaviest load setting: a processor that is mostly idle there is bigger than this load needs (or the load was too light to find its limit); one near 100% while missing your targets is too small. Memory notes come from the same source." /></h3>
        <RerunRecommended runIds={Array.from(new Set(g.points.map((p) => p.run_id)))} />
      </div>
      <div className="grid c2">
        {g.sizing.filter((z) => !shown || z.series === shown).map((z) => <SizingCard key={z.series} z={z} />)}
      </div>
      <div className="mt"><UtilizationCurves points={pts} /></div>
      <details className="mt"><summary className="small">All measurements in this group</summary>
        <table className="tbl mt">
          <thead><tr><th>Hardware · software · format</th><th className="num">Concurrent</th><th className="num">Batch</th><th className="num">{g.units.thr_label}</th><th className="num">{g.units.lat_label}</th><th className="num">median (ms)</th><th className="num">{g.units.cost_label}</th><th className="num">errors %</th><th className="num">busy %</th><th>Meets targets</th><th>Run</th></tr></thead>
          <tbody>{g.points.map((p, i) => <tr key={i}><td><Swatch style={styles[p.series + (p.batch > 1 ? ` ·b${p.batch}` : "")]} /><Badge kind={p.side}>{p.side.toUpperCase()}</Badge> {p.series.replace(/^(GPU|CPU) · /, "")}</td><td className="num">{p.conc}</td><td className="num">{p.batch > 1 ? p.batch : "–"}</td><td className="num">{fmt.n(p.throughput)}</td><td className="num">{fmt.n(p.latency)}</td><td className="num">{fmt.n(p.e2e_p50)}</td><td className="num">{fmt.usd(p.cost)}</td><td className="num">{fmt.n(p.error_pct, 1)}</td><td className="num" title={p.util?.busy_p95 != null ? `95th percentile ${fmt.n(p.util.busy_p95, 0)}%` : ""}>{p.util?.busy_mean != null ? `${fmt.n(p.util.busy_mean, 0)}%` : "–"}</td><td>{p.meets ? <Badge kind="ok">yes</Badge> : <Badge kind="bad">{p.reason || "no"}</Badge>}</td><td className="tiny">{p.run_id}</td></tr>)}</tbody>
        </table>
      </details>
    </div>
  );
}


const SIZING_KIND: Record<Sizing["verdict"], "ok" | "warn" | "bad" | "accent" | undefined> = { good: "ok", limit: "accent", over: "warn", under: "bad", unknown: undefined };

/** Copy a run to a new benchmark with every hardware option moved to its recommended size (one run at a time). */
function RerunRecommended({ runIds }: { runIds: string[] }) {
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (runIds.length !== 1) return null;
  const go = async () => {
    setBusy(true); setErr(null);
    try {
      const w = await api.wizardRecommended(runIds[0]);
      sessionStorage.setItem("wizard", JSON.stringify(w));
      nav("/new", { state: { step: 4 } });
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <div className="row">
      {err && <span className="small" style={{ color: "var(--bad)" }}>{err}</span>}
      <button className="btn sm primary" disabled={busy} onClick={go} title="Open a new benchmark with this run's settings and every hardware option set to the size its verdict recommends (kept where it already fits); nothing starts until you press Start">{busy ? "Opening…" : "Re-run with recommended sizes"}</button>
    </div>
  );
}

function SizingCard({ z }: { z: Sizing }) {
  const u = z.util_top;
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const sug = z.suggestion;
  // Open the wizard prefilled with this run's settings and the suggested size on this side, at the hardware
  // step; nothing starts until Start is pressed on the last step.
  const tryIt = async () => {
    if (!z.run_id || !sug?.instance_type) return;
    setBusy(true); setErr(null);
    try {
      const w = await api.wizardFrom(z.run_id);
      if (z.side === "gpu" && w.gpu) w.gpu = { ...w.gpu, instance_type: sug.instance_type, alternates: [] };
      if (z.side === "cpu") w.cpus = w.cpus.map((c) => (c.instance_type === z.instance_type ? { ...c, instance_type: sug.instance_type!, alternates: [] } : c));
      w.sized_for = w.models[0] ? `${w.models[0]}|${w.precision}` : w.sized_for;      // keep the suggested size; no fresh proposal over it
      w.copied_from = z.run_id;
      w.name = "";
      sessionStorage.setItem("wizard", JSON.stringify(w));
      nav("/new", { state: { step: 4 } });
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <div className={`tile ${z.side}`} style={{ cursor: "default" }}>
      <div className="t"><Badge kind={z.side}>{z.side.toUpperCase()}</Badge> {z.series.replace(/^(GPU|CPU) · /, "")} <Badge kind={SIZING_KIND[z.verdict]}>{z.label}</Badge></div>
      {z.size && <div className="d">{z.instance_type}: {z.size}</div>}
      {u != null && (
        <div className="row mt" style={{ gap: 14 }}>
          <Gauge value={u} />
          <div className="small">
            <div><b>{fmt.n(u, 0)}%</b> busy on average at the heaviest load ({z.load_top}){z.util_p95_top != null ? <>, <b>{fmt.n(z.util_p95_top, 0)}%</b> at its busiest</> : null}.</div>
            {z.util_low != null && z.points.length > 1 && <div className="muted">{fmt.n(z.util_low, 0)}% at the lightest load setting.</div>}
            <div className="muted">{z.with_samples} of {z.measurements} measurements had samples{z.busy_basis ? `; busy = ${z.busy_basis}` : ""}.</div>
          </div>
        </div>
      )}
      <div className="small mt">{z.text}</div>
      {sug && (
        <div className="mt" style={{ padding: "10px 12px", borderRadius: 10, background: "var(--bg-3)", border: "1px solid var(--line)" }}>
          <div className="row between">
            <div className="small" style={{ fontWeight: 650 }}>
              {sug.instance_type ? <>Suggested: <Badge kind={z.side}>{sug.instance_type}</Badge> {sug.kind === "keep" ? "(keep this size)" : sug.kind === "growth" ? "when traffic grows" : sug.kind === "smaller" ? "to spend less" : sug.kind === "stronger" ? "for a faster GPU" : "for more capacity"}</> : <>No better size in the catalog</>}
              {sug.kind !== "keep" && sug.usd_per_hour != null && sug.usd_delta_per_hour != null && (
                <span className="muted" style={{ fontWeight: 450 }}> · {fmt.usd(sug.usd_per_hour)}/h ({(sug.usd_delta_per_hour as number) < 0 ? "−" : "+"}{fmt.usd(Math.abs(sug.usd_delta_per_hour as number))}/h{sug.pct_of_current != null ? `, ${fmt.n(sug.pct_of_current, 0)}% of today's price` : ""})</span>
              )}
            </div>
            {sug.instance_type && z.run_id && sug.kind !== "growth" && sug.kind !== "keep" && <button className="btn sm primary" disabled={busy} onClick={tryIt} title="Open a new benchmark with this run's settings and this size on this side; nothing starts until you press Start">{busy ? "Opening…" : "Benchmark this size"}</button>}
          </div>
          <div className="small mt" style={{ marginTop: 6 }}>{sug.why}</div>
          {sug.pairing && <div className={"small mt " + (sug.pairing.fair ? "muted" : "")} style={{ marginTop: 6, color: sug.pairing.fair ? undefined : "var(--warn)" }}>{sug.pairing.text}</div>}
          {err && <div className="small mt" style={{ color: "var(--bad)" }}>{err}</div>}
        </div>
      )}
      {z.notes && z.notes.length > 0 && <ul className="small muted" style={{ margin: "6px 0 0", paddingLeft: 18 }}>{z.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
    </div>
  );
}

/** A small bar: 0–100% busy with the idle (<35%) and saturated (>80%) bands shaded. */
function Gauge({ value }: { value: number }) {
  const v = Math.max(0, Math.min(100, value));
  const color = v < 35 ? "var(--warn)" : v > 80 ? "var(--bad)" : "var(--ok)";
  return (
    <svg width="120" height="22" aria-label={`${Math.round(v)}% busy`}>
      <rect x="0" y="6" width="120" height="10" rx="5" fill="var(--bg-3)" stroke="var(--line)" />
      <rect x="0" y="6" width={42} height="10" rx="5" fill="var(--warn)" opacity="0.12" />
      <rect x="96" y="6" width={24} height="10" fill="var(--bad)" opacity="0.12" />
      <rect x="0" y="6" width={Math.max(2, (120 * v) / 100)} height="10" rx="5" fill={color} />
    </svg>
  );
}

/** Utilization over the whole run for every hardware option of the selected runs, from CloudWatch. */
function RunUtilization({ runIds }: { runIds: string[] }) {
  const [data, setData] = useState<Record<string, RunUtilization | { error: string }>>({});
  useEffect(() => {
    let alive = true;
    for (const id of runIds) {
      if (data[id]) continue;
      api.utilization(id).then((u) => alive && setData((d) => ({ ...d, [id]: u }))).catch((e) => alive && setData((d) => ({ ...d, [id]: { error: (e as Error).message } })));
    }
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runIds.join(",")]);
  return (
    <div className="card mt">
      <h2>Utilization over the run <Help text="CPU busy (whole host), host memory, GPU busy and GPU memory for each test machine across the whole run, read from CloudWatch (the agent on every host samples every 10 seconds; CloudWatch keeps 1-minute averages for 15 days, 5-minute ones for 63). Shaded bands are the measurements, labelled with their load setting (requests at a time, × inputs per request for batches); the gaps between them are model loading, warm-up and waiting. A machine that stays low inside the bands at the heaviest load is bigger than the load needs." /></h2>
      {runIds.map((id) => {
        const u = data[id];
        if (!u) return <Spinner key={id} label={`Reading CloudWatch for ${id}…`} />;
        if ("error" in u && !("targets" in u)) return <Note key={id} kind="bad">{id}: {u.error}</Note>;
        const run = u as RunUtilization;
        return (
          <div key={id} className="mt">
            <div className="row between"><h3 style={{ margin: 0 }}>{id}</h3><span className="tiny muted">{run.retention_note ?? run.reason}</span></div>
            {run.targets.length === 0 && <div className="small muted">{run.reason ?? "No hardware option recorded a measurement window."}</div>}
            <div className="grid c2 mt">
              {run.targets.map((t) => (
                <div key={t.target} className={`tile ${t.accelerator}`} style={{ cursor: "default" }}>
                  <div className="t"><Badge kind={t.accelerator}>{t.accelerator.toUpperCase()}</Badge> {t.instance_type ?? t.target}{t.spec ? <span className="muted small"> · {instanceLine(t.spec, true)}</span> : null}{t.scoped_to_run && t.instances > 1 ? <Badge kind="warn">{t.instances} instances (replaced during the run)</Badge> : null}{!t.scoped_to_run && t.hosts > 1 ? <Badge kind="warn" >{t.hosts} hosts published for this option in this window <Help text="The run did not record which instance served this hardware option, so the lines merge every host that reported under its id during the window: a replacement after a Spot interruption, or another run using the same option at the same time." /></Badge> : null}</div>
                  <UtilizationTimeline t={t} />
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ExecutiveSummary({ runIds, sla, limit, models, pricing, demandRpm }: { runIds: string[]; sla: Record<string, number | string | null> | null; limit: number; models: string[] | null; pricing: Pricing; demandRpm: number | null }) {
  const [model, setModel] = useState<{ name: string; call_id: string } | null>(null);
  const [res, setRes] = useState<ExecSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.execModel().then(setModel).catch((e) => setErr(e.message)); }, []);
  useEffect(() => { setRes(null); }, [runIds.join(","), JSON.stringify(sla), limit, JSON.stringify(models), JSON.stringify(pricing), demandRpm]);  // eslint-disable-line react-hooks/exhaustive-deps
  const go = async (force = false) => {
    setBusy(true); setErr(null);
    try { setRes(await api.execSummary({ run_ids: runIds, sla: sla ?? undefined, limit, models: models ?? undefined, pricing, demand_rpm: demandRpm, force })); }
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
