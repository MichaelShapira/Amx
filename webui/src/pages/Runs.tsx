import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type RunDetail, type RunRow } from "../api";
import { Badge, Help, Note, Spinner } from "../components/ui";

const statusKind = (s: string) => (s === "complete" || s === "done" || s === "ready" ? "ok" : s === "running" || s === "configuring" ? "accent" : s === "planned" || s === "stopped" ? undefined : s === "interrupted" || s === "incomplete" || s === "pending" ? "warn" : "bad");

export default function Runs() {
  const { id } = useParams();
  const nav = useNavigate();
  const [runs, setRuns] = useState<RunRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);

  const [deleting, setDeleting] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const load = () => api.runs().then((r) => { setRuns(r); setErr(null); }).catch((e) => setErr(e.message));
  useEffect(() => { load(); const t = setInterval(load, 30000); return () => clearInterval(t); }, []);
  const copyToNew = async (runId: string) => {
    setErr(null);
    try {
      const w = await api.wizardFrom(runId);
      sessionStorage.setItem("wizard", JSON.stringify(w));
      nav("/new", { state: { step: 6 } });
    } catch (e) { setErr(`Could not copy ${runId}: ${(e as Error).message}`); }
  };
  const remove = async (runId: string) => {
    if (!confirm(`Delete run ${runId}?\n\nThis removes its report, every measurement, the raw request records and the instance and capacity logs, from both storage and the run table. Other runs and the shared model cache are not touched. Instances are not affected.\n\nThis cannot be undone.`)) return;
    setDeleting(runId); setErr(null);
    try {
      const r = await api.deleteRun(runId);
      setNotice(`Deleted ${r.deleted}: ${r.objects} stored file${r.objects === 1 ? "" : "s"} and ${r.rows} table row${r.rows === 1 ? "" : "s"}${r.terminated_instances?.length ? `; ${r.terminated_instances.length} leftover instance${r.terminated_instances.length === 1 ? "" : "s"} terminated` : ""}.`);
      if (id === runId) nav("/runs");
      await load();
    } catch (e) { setErr((e as Error).message); }
    finally { setDeleting(null); }
  };
  const [detailErr, setDetailErr] = useState<string | null>(null);
  useEffect(() => {
    if (!id) { setDetail(null); setDetailErr(null); return; }
    let alive = true;
    setDetail(null); setDetailErr(null);
    const tick = () => api.run(id).then((d) => { if (alive) { setDetail(d); setDetailErr(null); } }).catch((e) => alive && setDetailErr(e.message));
    tick();
    const t = setInterval(tick, 15000);
    return () => { alive = false; clearInterval(t); };
  }, [id]);

  return (
    <div>
      <div className="topbar"><div><h1>Runs</h1><div className="muted">Every benchmark started from this tool or the command line. Pages refresh on their own.</div></div><Link className="btn primary" to="/new">New benchmark</Link></div>
      {err && <Note kind="bad">{err}</Note>}
      {notice && <Note kind="ok">{notice}</Note>}
      {!runs && !err && <Spinner label="Loading runs…" />}
      {runs && (
        <div className="card">
          <table className="tbl">
            <thead><tr><th>Run</th><th>Status <Help text="Running: in progress. Finished: the report is under Results. Refused by the pre-checks: nothing started (missing capacity, price, data or access); the reason is shown when you open the run." /></th><th>Grade <Help text="Decision-grade: at least 200 requests per measurement and the test repeated 3 times. Quick check: fewer; the report marks every number as indicative only (SMOKE)." /></th><th>Instances</th><th>Last update (UTC)</th><th></th></tr></thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} style={{ cursor: "pointer" }} onClick={() => nav(`/runs/${r.run_id}`)}>
                  <td>
                    <b>{r.name && r.name !== r.run_id ? r.name : r.run_id}</b>
                    {r.name && r.name !== r.run_id && <div className="tiny muted">{r.run_id}</div>}
                    {r.model && <div className="tiny muted">{r.model}</div>}
                    {r.errors && r.errors.length > 0 && (r.status === "failed" || r.status === "timed_out" || r.status === "refused") && <div className="small" style={{ color: "var(--bad)", maxWidth: 520, whiteSpace: "normal" }}>{r.errors[0]}</div>}
                  </td>
                  <td><Badge kind={r.not_run?.length ? "bad" : statusKind(r.status)}>{r.status_label}</Badge></td>
                  <td>{r.purchase ? <Badge kind={r.grade === "full" ? "ok" : "warn"}>{r.grade === "full" ? "Decision-grade" : "Quick check"}</Badge> : <span className="muted">–</span>}</td>
                  <td>{r.purchase ?? "–"}</td>
                  <td className="muted">{r.updated_at ?? "–"}</td>
                  <td>
                    <div className="row" style={{ justifyContent: "flex-end", gap: 6 }}>
                      {r.status === "complete" && <Link className="btn sm" to={`/results?runs=${r.run_id}`} onClick={(e) => e.stopPropagation()}>Results →</Link>}
                      <button className="btn sm" title="Open a new benchmark prefilled with this run's settings; nothing starts until you press Start" onClick={(e) => { e.stopPropagation(); copyToNew(r.run_id); }}>Copy to new run</button>
                      <button className="btn sm danger" disabled={deleting === r.run_id || r.status === "running" || r.status === "planned"}
                        title={r.status === "running" || r.status === "planned" ? "Stop the run first" : "Delete this run and everything it stored"}
                        onClick={(e) => { e.stopPropagation(); remove(r.run_id); }}>{deleting === r.run_id ? "Deleting…" : "Delete"}</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {runs.length === 0 && <div className="muted small mt">No runs yet.</div>}
        </div>
      )}
      {id && (
        <div className="card mt">
          <div className="row between">
            <div>
              <h2 style={{ margin: 0 }}>{detail?.name && detail.name !== id ? detail.name : id}</h2>
              {detail?.name && detail.name !== id && <div className="tiny muted">{id}</div>}
            </div>
            <div className="row">
              {detail?.has_report && <Link className="btn primary sm" to={`/results?runs=${id}`}>Open results →</Link>}
              {detail?.status === "running" && <button className="btn danger sm" onClick={async () => { if (confirm("Stop this run? Its instances are terminated right away; finished measurements are kept.")) { const r = await api.stop(id); setNotice(`Stopped ${id}: ${r.terminated_instances.length} instance${r.terminated_instances.length === 1 ? "" : "s"} terminated.`); } }}>Stop run</button>}
              {detail && <button className="btn sm" title="Open a new benchmark prefilled with this run's settings; nothing starts until you press Start" onClick={() => copyToNew(id)}>Copy to new run</button>}
              {detail && detail.status !== "running" && detail.status !== "planned" && <button className="btn danger sm" disabled={deleting === id} onClick={() => remove(id)}>{deleting === id ? "Deleting…" : "Delete run"}</button>}
            </div>
          </div>
          {!detail && !detailErr && <Spinner label="Loading the run…" />}
          {!detail && detailErr && <Note kind="bad"><b>Nothing to show for this run.</b> {detailErr} <Link to="/runs">Back to the list</Link>.</Note>}
          {detail && (
            <>
              <div className="row mt small">
                <Badge kind={detail.failure ? "bad" : detail.targets.some((t) => t.did_not_run) ? "bad" : statusKind(detail.status)}>{detail.status_label}</Badge>
                {detail.purchase && <Badge kind={detail.grade === "full" ? "ok" : "warn"}>{detail.grade === "full" ? "Decision-grade" : "Quick check"}</Badge>}
                {detail.progress.total > 0 && <span className="muted">{detail.progress.done} of {detail.progress.total} measurements done</span>}
                {detail.status === "planned" && <span className="muted">Starting: the pre-checks are running (prices, images, data); this takes about a minute.</span>}
              </div>
              {detail.progress.total > 0 && <div className="progress mt"><div style={{ width: `${(100 * detail.progress.done) / Math.max(1, detail.progress.total)}%` }} /></div>}
              {detail.refusal && <Note kind="bad" ><b>Refused by the pre-checks:</b> {detail.refusal.join("; ")}</Note>}
              {detail.failure && <Note kind="bad"><b>This run produced nothing.</b> {detail.failure.join(" ")}</Note>}
              {detail.notices?.map((n, i) => <Note key={"n" + i} kind={n.kind}><b>{n.title}.</b> {n.text}</Note>)}
              {detail.targets.length > 0 && (
                <>
                  <h3 className="mt">Instances <Help text="One test instance per hardware option. The state comes live from EC2; the market says whether it is a Spot or an On-Demand instance and, for On-Demand, whether it was a fallback after the Spot waiting time ran out. Capacity refusals and Spot interruptions seen during this run are counted. 'Did not run' means AWS never granted a machine within the attempt limit set when the run was started." /></h3>
                  <table className="tbl">
                    <thead><tr><th>Hardware option</th><th>Instance</th><th>State</th><th>Market</th><th>Status</th></tr></thead>
                    <tbody>{detail.targets.map((t) => (
                      <tr key={t.hardware}>
                        <td><Badge kind={t.hardware_label.startsWith("GPU") ? "gpu" : "cpu"}>{t.hardware_label.startsWith("GPU") ? "GPU" : "CPU"}</Badge> {t.hardware_label.replace(/^(GPU|CPU) · /, "")}</td>
                        <td className="small">{t.instance_type ?? "–"}{t.instance_id ? <div className="tiny">{t.instance_id}</div> : null}</td>
                        <td>{t.did_not_run ? <Badge kind="bad">Did not run</Badge> : t.state ? <Badge kind={t.state === "running" ? "ok" : t.state === "pending" ? "accent" : t.state === "gone" || t.state === "terminated" || t.state === "stopped" ? undefined : "warn"}>{t.state === "gone" ? "Gone" : t.state_label}</Badge> : <Badge kind="warn">{t.capacity_refusals ? `No capacity yet (${t.capacity_refusals}${t.attempts_max ? ` of ${t.attempts_max}` : ""} refused)` : "Not launched"}</Badge>}</td>
                        <td>{t.lifecycle ? <Badge kind={t.lifecycle === "spot" ? "accent" : t.fallback_from_spot ? "warn" : undefined}>{t.lifecycle === "spot" ? "Spot" : t.fallback_from_spot ? "On-Demand (fallback from Spot)" : "On-Demand"}</Badge> : <span className="muted">–</span>}</td>
                        <td className="small">{t.status}{t.events.length > 0 && <details className="tiny"><summary>history</summary>{t.events.map((e, i) => <div key={i}>{e.time?.slice(11, 19)} {e.event}{e.instance_type ? ` ${e.instance_type}` : ""}{e.lifecycle ? ` (${e.lifecycle})` : ""}{e.state_reason ? ` · ${e.state_reason}` : ""}</div>)}</details>}</td>
                      </tr>))}</tbody>
                  </table>
                </>
              )}
              {detail.cells.length > 0 && (<>
              <h3 className="mt">Models on hardware options <Help text="One row per model served on one hardware option. Models on the same instance run one after another: Waiting for the instance → Loading the model → Measuring → Done. 'Queued behind X' means another model of this run is using the instance right now. 'Not run — no capacity' means AWS never granted a machine for that hardware option; the reason is in the last column." /></h3>
              <table className="tbl">
                <thead><tr><th>Model</th><th>Hardware</th><th>Format</th><th>Software</th><th>Status</th><th className="num">Measurements</th><th>Why</th></tr></thead>
                <tbody>{detail.cells.map((c, i) => <tr key={i}><td>{c.model}</td><td>{c.hardware_label}</td><td>{c.precision}</td><td>{c.engine}</td><td><Badge kind={statusKind(c.status)}>{c.status_label}</Badge></td><td className="num">{c.status === "not_run" ? "0" : c.measurements}</td><td className={"small " + (c.status === "not_run" ? "" : "muted")} style={{ maxWidth: 420 }}>{c.error}</td></tr>)}</tbody>
              </table>
              <h3 className="mt">Measurements <Help text="One row per measurement: one load setting (requests at a time), one repetition, one model, one hardware option. 'Requests done' counts the requests sent so far; the accuracy pass of the detection use case classifies every test document once." /></h3>
              <table className="tbl">
                <thead><tr><th>Model</th><th>Hardware</th><th>What</th><th className="num">Concurrent</th><th className="num">Batch</th><th className="num">Pass</th><th>Status</th><th className="num">Requests done</th><th className="num">Errors</th></tr></thead>
                <tbody>{detail.levels.map((l, i) => <tr key={i}><td>{l.model}</td><td className="small">{l.hardware_label}</td><td>{l.what}</td><td className="num">{l.conc ?? "–"}</td><td className="num">{l.batch > 1 ? l.batch : "–"}</td><td className="num">{l.pass ?? "–"}</td><td><Badge kind={statusKind(l.status)}>{l.status_label}</Badge>{l.detail && <div className="tiny">{l.detail}</div>}</td><td className="num">{l.completed ?? "–"}</td><td className="num">{l.errors ?? "–"}</td></tr>)}</tbody>
              </table>
              </>)}
            </>
          )}
        </div>
      )}
    </div>
  );
}
