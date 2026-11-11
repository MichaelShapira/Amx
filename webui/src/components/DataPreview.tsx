import { useEffect, useState } from "react";
import { api, type UseCase } from "../api";
import { useCatalog } from "../App";
import { Badge, Help, Spinner } from "./ui";

/** What the models are asked: counts per set or scenario, then real examples with expected answers. */
export default function DataPreview({ useCase, promptSet, workloads, compact, model }: {
  useCase: UseCase; promptSet?: string; workloads?: string[]; compact?: boolean; model?: string;
}) {
  const cat = useCatalog();
  const [info, setInfo] = useState<any>(null);
  const [items, setItems] = useState<any[] | null>(null);
  const [seed, setSeed] = useState(0);
  const [wl, setWl] = useState<string>(workloads?.[0] ?? "W1");
  const [label, setLabel] = useState("");
  const [hard, setHard] = useState("");
  const [mid, setMid] = useState(model ?? "m1-qwen2.5-3b");
  const [ps, setPs] = useState(promptSet ?? "short");

  useEffect(() => { api.data(useCase).then(setInfo); }, [useCase]);
  useEffect(() => { if (promptSet) setPs(promptSet); }, [promptSet]);
  useEffect(() => { if (workloads && !workloads.includes(wl)) setWl(workloads[0] ?? "W1"); }, [workloads, wl]);
  useEffect(() => {
    setItems(null);
    const q = useCase === "standard" ? { prompt_set: ps, model: mid, n: compact ? 3 : 5, seed }
      : { workload: wl, label: label || undefined, hard_case: hard || undefined, n: compact ? 3 : 5, seed };
    api.sample(useCase, q).then(setItems).catch(() => setItems([]));
  }, [useCase, ps, mid, wl, label, hard, seed, compact]);

  if (!info) return <Spinner label="Loading the test data…" />;
  if (!info.available) return <div className="note warn">{info.reason}</div>;

  const stdModels = cat?.models.filter((m) => m.use_cases.includes("standard")) ?? [];
  return (
    <div>
      {!compact && <p className="muted">{info.summary}</p>}
      {useCase === "standard" ? (
        <>
          {!compact && (
            <table className="tbl mb">
              <thead><tr><th>Prompt set</th><th className="num">Prompts</th><th>Length (tokens)</th><th>Used for</th></tr></thead>
              <tbody>
                {Object.entries(info.sets as Record<string, any>).map(([k, s]) => {
                  const m = s.models[mid] ?? Object.values(s.models)[0] as any;
                  return <tr key={k}><td><b>{s.label}</b></td><td className="num">{m?.n}</td><td>{m?.min_tokens}–{m?.max_tokens} (target {m?.target})</td><td className="muted">{s.help}</td></tr>;
                })}
              </tbody>
            </table>
          )}
          <div className="row small mb">
            {!promptSet && (
              <label className="row">Set <select value={ps} onChange={(e) => setPs(e.target.value)}>
                {Object.entries(info.sets as Record<string, any>).map(([k, s]) => <option key={k} value={k}>{s.label}</option>)}
              </select></label>
            )}
            <label className="row">Shown in the chat format of
              <select value={mid} onChange={(e) => setMid(e.target.value)}>
                {stdModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
              <Help text="Prompts are wrapped in each model's own chat template, so the exact text differs slightly per model while the content is the same." />
            </label>
            <button className="btn sm" onClick={() => setSeed(seed + 1)}>Other examples</button>
          </div>
        </>
      ) : (
        <>
          {!compact && (
            <table className="tbl mb">
              <thead><tr><th>Scenario</th><th className="num">Documents</th><th className="num">Sensitive</th><th className="num">Hard negatives <Help text="Documents that look sensitive but are not: masked card numbers, documentation placeholders, public business data, first names alone. The report shows false positives per type." /></th><th>Expected answers</th></tr></thead>
              <tbody>
                {Object.entries(info.workloads as Record<string, any>).filter(([k]) => !workloads || workloads.includes(k)).map(([k, w]) => (
                  <tr key={k}><td><b>{w.label}</b><div className="tiny">{w.help}{w.note ? " " + w.note : ""}</div></td><td className="num">{w.documents}</td><td className="num">{w.positive}</td><td className="num">{w.hard_negatives}</td><td>{w.labels.map((l: string) => <Badge key={l}>{l}</Badge>)}</td></tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="row small mb">
            <label className="row">Scenario <select value={wl} onChange={(e) => { setWl(e.target.value); setLabel(""); setHard(""); }}>
              {Object.entries(info.workloads as Record<string, any>).filter(([k]) => !workloads || workloads.includes(k)).map(([k, w]) => <option key={k} value={k}>{w.label}</option>)}
            </select></label>
            <label className="row">Expected answer <select value={label} onChange={(e) => setLabel(e.target.value)}>
              <option value="">any</option>{(info.workloads[wl]?.labels ?? []).map((l: string) => <option key={l} value={l}>{l}</option>)}
            </select></label>
            <label className="row">Hard negative <select value={hard} onChange={(e) => setHard(e.target.value)}>
              <option value="">any</option><option value="none">none</option>{(info.workloads[wl]?.hard_cases ?? []).map((l: string) => <option key={l} value={l}>{l}</option>)}
            </select></label>
            <button className="btn sm" onClick={() => setSeed(seed + 1)}>Other examples</button>
            <span className="tiny">Dataset {info.version}, fingerprint {info.fingerprint}</span>
          </div>
        </>
      )}
      {items == null ? <Spinner /> : items.length === 0 ? <div className="note">No documents match.</div> : items.map((x) => (
        <details key={x.id} className="doc">
          <summary>
            <span className="hd" style={{ display: "inline-flex" }}>
              {x.label && <Badge kind={x.label === "SENSITIVE" || x.label === "BLOCK" || (x.label !== "NOT_SENSITIVE" && x.label !== "ALLOW" && x.label !== "NONE") ? "bad" : "ok"}>expected: {x.label}</Badge>}
              {x.categories?.length ? <Badge>{x.categories.join(", ")}</Badge> : null}
              {x.hard_case && <Badge kind="warn">hard negative: {x.hard_case}</Badge>}
              <span className="muted">{x.id} · about {x.tokens} tokens{x.bytes ? ` · ${x.bytes} bytes` : ""}{x.source ? ` · source ${x.source}` : ""}</span>
            </span>
          </summary>
          <pre style={{ marginTop: 8 }}>{x.text}</pre>
          {x.values?.length ? <div className="small muted">Sensitive values in this document: {x.values.map((v: any, i: number) => <span key={i}><b>{v.category}</b>: {v.text}{i < x.values.length - 1 ? "; " : ""}</span>)}</div> : null}
        </details>
      ))}
      {!compact && info.sources && (
        <div className="tiny mt">Sources: {info.sources.map((s: any) => `${s.name} (${s.role}; ${s.license})`).join(" · ")}</div>
      )}
      {!compact && info.policy && (
        <details className="mt"><summary>The classification policy the models are given</summary><pre>{info.policy}</pre></details>
      )}
    </div>
  );
}
