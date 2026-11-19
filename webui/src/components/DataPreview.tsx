import { useEffect, useState } from "react";
import { api, type UseCase } from "../api";
import { useCatalog } from "../App";
import { Badge, Help, Spinner } from "./ui";

/** What the models are asked: counts per set or scenario, then real examples with expected answers. */
export default function DataPreview({ useCase, promptSet, workloads, directions, compact, model }: {
  useCase: UseCase; promptSet?: string; workloads?: string[]; directions?: string[]; compact?: boolean; model?: string;
}) {
  const cat = useCatalog();
  const [info, setInfo] = useState<any>(null);
  const [items, setItems] = useState<any[] | null>(null);
  const [seed, setSeed] = useState(0);
  const [wl, setWl] = useState<string>(workloads?.[0] ?? "W1");
  const [label, setLabel] = useState("");
  const [hard, setHard] = useState("");
  const [mid, setMid] = useState(model ?? "m1-qwen3-1.7b");
  const [ps, setPs] = useState(promptSet ?? (useCase === "translation" || useCase === "embedding" ? "sentence" : useCase === "intel" ? "128" : "short"));
  const [lang, setLang] = useState<string>("");
  const [dir, setDir] = useState<string>(directions?.[0] ?? "en-de");

  // Clear the previous use case's description before loading the next one: rendering the
  // detection view with the chat description (or the reverse) crashed on a missing field.
  useEffect(() => { setInfo(null); setItems(null); api.data(useCase).then(setInfo).catch(() => setInfo({ available: false, reason: "the API is not reachable" })); }, [useCase]);
  useEffect(() => { if (promptSet) setPs(promptSet); }, [promptSet]);
  useEffect(() => { if (workloads && !workloads.includes(wl)) setWl(workloads[0] ?? "W1"); }, [workloads, wl]);
  useEffect(() => { if (directions && !directions.includes(dir)) setDir(directions[0] ?? "en-de"); }, [directions, dir]);
  useEffect(() => { if (!promptSet) setPs(useCase === "translation" || useCase === "embedding" ? "sentence" : useCase === "intel" ? "128" : "short"); }, [useCase, promptSet]);
  useEffect(() => {
    setItems(null);
    const q = useCase === "standard" ? { prompt_set: ps, model: mid, n: compact ? 3 : 5, seed }
      : useCase === "translation" ? { prompt_set: ps, direction: dir, n: compact ? 2 : 4, seed }
      : useCase === "embedding" ? { prompt_set: ps, direction: lang || undefined, n: compact ? 2 : 3, seed }
      : useCase === "intel" ? { prompt_set: ps, model: mid, n: compact ? 2 : 3, seed }
      : { workload: wl, label: label || undefined, hard_case: hard || undefined, n: compact ? 3 : 5, seed };
    api.sample(useCase, q).then(setItems).catch(() => setItems([]));
  }, [useCase, ps, mid, wl, dir, lang, label, hard, seed, compact]);

  if (!info) return <Spinner label="Loading the test data…" />;
  if (!info.available) return <div className="note warn">{info.reason}</div>;
  const expectedKind = useCase === "standard" ? "prompts" : useCase === "translation" ? "parallel" : useCase === "embedding" ? "embedding" : useCase === "intel" ? "intel" : "documents";
  const intelModels = cat?.models.filter((m) => m.use_cases.includes("intel")) ?? [];
  if (info.kind !== expectedKind) return <Spinner label="Loading the test data…" />;

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
      ) : useCase === "embedding" ? (
        <>
          {!compact && (
            <div className="small muted mb">Corpus {info.version}, fingerprint {info.fingerprint} · {info.items} items × 3 languages · prefix on every text: <code>{info.prefix}</code></div>
          )}
          <div className="row small mb">
            {!promptSet && (
              <label className="row">Text size <select value={ps} onChange={(e) => setPs(e.target.value)}>
                {Object.entries(info.sizes as Record<string, any>).map(([k, z]) => <option key={k} value={k}>{z.label}</option>)}
              </select></label>
            )}
            <label className="row">Language <select value={lang} onChange={(e) => setLang(e.target.value)}><option value="">all three</option><option value="en">English</option><option value="de">German</option><option value="es">Spanish</option></select></label>
            <button className="btn sm" onClick={() => setSeed(seed + 1)}>Other examples</button>
          </div>
        </>
      ) : useCase === "intel" ? (
        <>
          {!compact && info.published && (
            <table className="tbl mb">
              <thead><tr><th>Model (Intel's figures)</th><th>System</th><th className="num">Cores</th><th>Shape</th><th className="num">Concurrency</th><th className="num">tok/s</th><th className="num">tok/s per core</th><th className="num">TPOT ms</th></tr></thead>
              <tbody>
                {Object.entries(info.published as Record<string, any[]>).flatMap(([m, rows]) => rows.filter((r) => r.io === `${ps}/128` || r.io === `${ps}/2048`).slice(0, 6).map((r, i) => (
                  <tr key={m + i}><td>{i === 0 ? <b>{m}</b> : ""}</td><td className="small">{r.sockets}× {r.cpu_model}</td><td className="num">{r.total_cores}</td><td>{r.io}</td><td className="num">{r.max_concurrency}</td><td className="num">{Math.round(r.throughput_tok_s)}</td><td className="num">{(r.throughput_tok_s / r.total_cores).toFixed(1)}</td><td className="num">{Math.round(r.tpot_ms)}</td></tr>
                )))}
              </tbody>
            </table>
          )}
          <div className="row small mb">
            {!promptSet && (
              <label className="row">Prompt length <select value={ps} onChange={(e) => setPs(e.target.value)}><option value="128">128 tokens</option><option value="2048">2,048 tokens</option></select></label>
            )}
            <label className="row">Tokenizer of <select value={intelModels.some((m) => m.id === mid) ? mid : intelModels[0]?.id ?? mid} onChange={(e) => setMid(e.target.value)}>
              {intelModels.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select></label>
            <button className="btn sm" onClick={() => setSeed(seed + 1)}>Other examples</button>
          </div>
        </>
      ) : useCase === "translation" ? (
        <>
          {!compact && (
            <table className="tbl mb">
              <thead><tr><th>Text size</th><th className="num">Items</th><th>Sentences</th><th>Words (English)</th><th>Answer limit</th><th>Used for</th></tr></thead>
              <tbody>
                {Object.entries(info.sizes as Record<string, any>).map(([k, z]) => (
                  <tr key={k}><td><b>{z.label}</b></td><td className="num">{z.items}</td><td>{z.sentences?.[0]}–{z.sentences?.[1]}</td><td>{z.words_en?.[0]}–{z.words_en?.[1]}</td><td>{z.answer_tokens} tokens</td><td className="muted">{z.help}</td></tr>
                ))}
              </tbody>
            </table>
          )}
          {!compact && (
            <div className="small muted mb">Corpus {info.version}, fingerprint {info.fingerprint} · {info.items} items · domains: {Object.entries(info.domains as Record<string, number>).map(([d, n]) => `${d} (${n})`).join(", ")} · characters: {Object.entries(info.chars as Record<string, number>).map(([l, n]) => `${l.toUpperCase()} ${n.toLocaleString()}`).join(", ")}</div>
          )}
          <div className="row small mb">
            {!promptSet && (
              <label className="row">Text size <select value={ps} onChange={(e) => setPs(e.target.value)}>
                {Object.entries(info.sizes as Record<string, any>).map(([k, z]) => <option key={k} value={k}>{z.label}</option>)}
              </select></label>
            )}
            <label className="row">Direction <select value={dir} onChange={(e) => setDir(e.target.value)}>
              {Object.entries(info.directions as Record<string, any>).filter(([k]) => !directions || directions.includes(k)).map(([k, d]) => <option key={k} value={k}>{d.label}</option>)}
            </select></label>
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
      {items == null ? <Spinner /> : items.length === 0 ? <div className="note">No documents match.</div> : useCase === "translation" ? items.map((x) => (
        <details key={x.id} className="doc" open={compact ? undefined : true}>
          <summary><span className="hd" style={{ display: "inline-flex" }}><Badge>{cat?.directions?.[x.direction]?.label ?? x.direction}</Badge><span className="muted">{x.id} · {x.domain} · {x.sentences} sentence{x.sentences === 1 ? "" : "s"} · {x.words} words · {x.chars} characters</span></span></summary>
          <div className="grid c2 mt" style={{ gap: 12 }}>
            <div><div className="tiny muted">What the model receives</div><pre style={{ marginTop: 4 }}>{x.text}</pre></div>
            <div><div className="tiny muted">Reference translation (scored against)</div><pre style={{ marginTop: 4 }}>{x.reference}</pre></div>
          </div>
        </details>
      )) : useCase === "embedding" ? items.map((x) => (
        <details key={x.id} className="doc" open={compact ? undefined : true}>
          <summary><span className="hd" style={{ display: "inline-flex" }}><Badge>{{ en: "English", de: "German", es: "Spanish" }[x.lang as string] ?? x.lang}</Badge><span className="muted">{x.id} · {x.domain} · {x.words} words · {x.chars} characters</span></span></summary>
          <pre style={{ marginTop: 8 }}>{x.text}</pre>
        </details>
      )) : useCase === "intel" ? items.map((x) => (
        <details key={x.id} className="doc">
          <summary><span className="hd" style={{ display: "inline-flex" }}><Badge>random tokens</Badge><span className="muted">{x.id} · {x.tokens} tokens</span></span></summary>
          <pre style={{ marginTop: 8 }}>{x.text}</pre>
        </details>
      )) : items.map((x) => (
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
