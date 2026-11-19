// Setup: the Hugging Face token (kept in AWS Secrets Manager), the models the benchmark offers
// (added from and removed to the live catalog in the bench bucket) and the Bedrock model that
// reasons about hardware and writes summaries. Changes need the operators group; everyone can look.
import { useEffect, useRef, useState } from "react";
import { api, type OnboardJob, type Setup as SetupData, type SetupModel } from "../api";
import { useSession } from "../components/AuthGate";
import { Badge, Help, Note, Spinner } from "../components/ui";

export default function Setup() {
  const session = useSession();
  const canEdit = session.operator;
  const [data, setData] = useState<SetupData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const load = () => api.setup().then((d) => { setData(d); setErr(null); return d; }).catch((e) => { setErr(e.message); return null; });
  useEffect(() => { void load(); }, []);
  return (
    <div>
      <div className="topbar"><div><h1>Setup</h1><div className="muted">What this deployment can reach and which models it offers. Nothing here starts a benchmark.</div></div></div>
      {!canEdit && <Note kind="warn">You are signed in as a viewer: you can see these settings but not change them. Ask an operator to make changes.</Note>}
      {err && <Note kind="bad">{err}</Note>}
      {notice && <Note kind="ok">{notice}</Note>}
      {!data && !err && <Spinner label="Loading settings…" />}
      {data && (
        <>
          <TokenCard data={data} canEdit={canEdit} onSaved={(m) => { setNotice(m); void load(); }} onError={setErr} />
          <ModelsCard data={data} canEdit={canEdit} reload={load} onNotice={setNotice} onError={setErr} />
          <ReasoningCard data={data} canEdit={canEdit} onSaved={(m) => { setNotice(m); void load(); }} onError={setErr} />
        </>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------- Hugging Face token

function TokenCard({ data, canEdit, onSaved, onError }: { data: SetupData; canEdit: boolean; onSaved: (m: string) => void; onError: (m: string) => void }) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const st = data.hf_token;
  const save = async () => {
    setBusy(true);
    try { await api.setHfToken(token); setToken(""); onSaved("Hugging Face token saved. The benchmark hosts read it from AWS Secrets Manager when they download gated models."); }
    catch (e) { onError((e as Error).message); }
    finally { setBusy(false); }
  };
  return (
    <div className="card mb">
      <h2>Hugging Face token <Help text="Some models on Hugging Face (Gemma, Llama, Mistral…) are gated: you accept their license on huggingface.co and then download them with a token from your account. The token is stored in AWS Secrets Manager, read by the benchmark hosts and by the model-adding job, and never shown again here." /></h2>
      <div className="row mb">
        {st.available
          ? st.configured
            ? <Badge kind="ok">Set · last changed {st.last_changed}</Badge>
            : <Badge kind="warn">Not set — gated models cannot be downloaded</Badge>
          : <Badge kind="bad">Unavailable: {st.reason}</Badge>}
        <span className="tiny">{st.name ? `Secret: ${st.name}` : ""}</span>
      </div>
      {canEdit && st.available && (
        <div className="row">
          <input type="password" autoComplete="off" placeholder="hf_…" value={token} onChange={(e) => setToken(e.target.value)} style={{ width: 420 }} />
          <button className="btn primary" disabled={busy || !token.trim()} onClick={save}>{busy ? "Saving…" : st.configured ? "Replace token" : "Save token"}</button>
          <span className="tiny">Hugging Face → Settings → Access Tokens. A read token is enough.</span>
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------- models

function ModelsCard({ data, canEdit, reload, onNotice, onError }: {
  data: SetupData; canEdit: boolean; reload: () => Promise<SetupData | null>; onNotice: (m: string) => void; onError: (m: string) => void;
}) {
  const [repo, setRepo] = useState("");
  const [job, setJob] = useState<OnboardJob | null>(data.running_job);
  const [removing, setRemoving] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  // Poll the add job until it ends; the catalog is reloaded when it does.
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const tick = async () => {
      try {
        const j = await api.addModelJob(job.id);
        setJob(j);
        if (j.status !== "running") { await reload(); if (j.status === "done") onNotice(`${j.model?.name ?? j.repo} added. Its first benchmark run is the real test of which serving software loads it.`); }
      } catch (e) { onError((e as Error).message); }
    };
    timer.current = window.setTimeout(tick, 2000);
    return () => { if (timer.current) window.clearTimeout(timer.current); };
  }, [job]);
  const add = async () => {
    try { const r = await api.addModel(repo.trim()); setRepo(""); setJob({ id: r.job_id, repo: repo.trim(), status: "running", stage: "Starting", elapsed_s: 0, notes: [], error: null, model: null }); }
    catch (e) { onError((e as Error).message); }
  };
  const remove = async (m: SetupModel) => {
    if (!confirm(`Remove ${m.name} from the list of models this benchmark offers?\n\nFinished runs keep their results. You can add it again later.`)) return;
    setRemoving(m.id);
    try { await api.removeModel(m.id); await reload(); onNotice(`${m.name} removed.`); }
    catch (e) { onError((e as Error).message); }
    finally { setRemoving(null); }
  };
  const shipped = data.models.filter((m) => !m.user_added);
  const added = data.models.filter((m) => m.user_added);
  return (
    <div className="card mb">
      <h2>Models <Help text="The Hugging Face models a new benchmark can pick from. Removing one only takes it off the list; adding one prepares its test prompts with the model's own tokenizer, which takes a few minutes." /></h2>
      <Note kind="warn">
        <b>What this page can and cannot promise.</b> When you add a model, the benchmark reads what kind of model it is and prepares its
        test prompts. It cannot know for sure that the serving software (vLLM, OpenVINO, PyTorch) will load it: each supports a list of model
        families, and a new family may fail at start. Models marked <Badge kind="ok">proven in a run</Badge> have finished a measurement here;
        the others are <i>expected</i> to work. The first run is the real test, and the run page names the cause if a model does not start.
      </Note>
      {canEdit && (
        <div className="row mt mb">
          <input placeholder="owner/model-name, e.g. mistralai/Mistral-7B-Instruct-v0.3" value={repo} onChange={(e) => setRepo(e.target.value)}
                 onKeyDown={(e) => { if (e.key === "Enter" && repo.trim() && job?.status !== "running") void add(); }} style={{ width: 460 }} disabled={job?.status === "running"} />
          <button className="btn primary" disabled={!repo.trim() || job?.status === "running"} onClick={add}>Add model</button>
          <span className="tiny">Chat models, encoder-decoder instruction models, text classifiers and token detectors. Gated models need the token above.</span>
        </div>
      )}
      {data.proven_error && <Note kind="warn">The run history could not be read ({data.proven_error}), so every model shows "not yet run" for now.</Note>}
      {job && <JobCard job={job} onDismiss={() => setJob(null)} />}
      <ModelTable title="Added here" models={added} canEdit={canEdit} removing={removing} onRemove={remove} empty="No models added yet." />
      <ModelTable title="Shipped with the benchmark" models={shipped} canEdit={canEdit} removing={removing} onRemove={remove} empty="" />
      <div className="tiny mt">Catalog: {data.catalog_source === "seed" ? "the built-in list (nothing changed yet)" : data.catalog_source ?? "–"}</div>
    </div>
  );
}

function JobCard({ job, onDismiss }: { job: OnboardJob; onDismiss: () => void }) {
  const kind = job.status === "running" ? undefined : job.status === "done" ? "ok" : "bad";
  return (
    <Note kind={kind}>
      <div className="row between">
        <div>
          <b>{job.status === "running" ? "Adding" : job.status === "done" ? "Added" : "Could not add"} {job.repo}</b>
          {job.status === "running" && <span className="muted"> · {job.stage}… ({Math.round(job.elapsed_s)} s)</span>}
          {job.status === "done" && job.pools && <span className="muted"> · {job.pools.prompt_files} prompt files{job.pools.dspm_files ? `, ${job.pools.dspm_files} sensitive-data files` : ""}</span>}
        </div>
        {job.status !== "running" && <button className="btn sm" onClick={onDismiss}>Dismiss</button>}
      </div>
      {job.status === "running" && <div className="progress mt"><div style={{ width: `${Math.min(95, 5 + job.elapsed_s / 3)}%` }} /></div>}
      {job.error && <div className="mt">{job.error}</div>}
      {job.model && <div className="tiny mt">{job.model.type} · {job.model.params_b} B parameters · license {job.model.license}{job.model.gated ? " · gated" : ""}</div>}
      {job.notes.length > 0 && <ul className="small mt" style={{ margin: "8px 0 0", paddingLeft: 18 }}>{job.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
    </Note>
  );
}

function ModelTable({ title, models, canEdit, removing, onRemove, empty }: {
  title: string; models: SetupModel[]; canEdit: boolean; removing: string | null; onRemove: (m: SetupModel) => void; empty: string;
}) {
  if (models.length === 0 && !empty) return null;
  return (
    <div className="mt">
      <div className="label">{title}</div>
      {models.length === 0 ? <div className="tiny">{empty}</div> : (
        <table className="tbl">
          <thead><tr>
            <th>Model<Help text="The model's name, its Hugging Face repository and the exact version (revision) every run uses, so results stay comparable over time. 'gated' = Hugging Face requires an accepted license and a token to download it." /></th>
            <th>Kind<Help text="What the model does: a chat model writes text; a classifier gives a label; a detector marks words; an embedding model turns text into vectors for search. The kind decides which use cases and targets apply." /></th>
            <th>Size<Help text="Billions of parameters. Size drives memory needs and speed: a 4 B model needs about 8 GB in BF16 and is read from memory for every generated word." /></th>
            <th>License<Help text="The model's license. A research-only license means its results are for evaluation, not for a commercial decision." /></th>
            <th>Serving software<Help text="The engines that can run this model here (vLLM, OpenVINO, PyTorch). 'proven in a run' = a benchmark with this model on that engine has finished in this deployment, so it is known to load and answer. 'not yet run' = expected to work from the model's family, but not yet demonstrated here; the first run is the test, and the run page names the cause if the engine refuses the model. This is shown so you know which combinations carry that first-run risk." /></th>
            <th>Suites<Help text="The use cases this model can be benchmarked in (chat and classification, sensitive-data detection, translation, embeddings, Intel's published benchmark)." /></th>
            <th />
          </tr></thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id}>
                <td><b>{m.name}</b><div className="tiny"><a href={`https://huggingface.co/${m.repo}`} target="_blank" rel="noreferrer">{m.repo}</a> @ {m.revision}{m.gated ? " · gated" : ""}</div></td>
                <td>{m.kind}</td>
                <td>{m.params_b} B</td>
                <td>{m.license}</td>
                <td>
                  <div className="row" style={{ gap: 6 }}>
                    {m.engines.map((e) => <span key={e.id} title={e.proven ? `${e.label} has served this model in a finished benchmark here.` : `${e.label} is expected to run this model (its family is supported) but has not done so in a finished benchmark here yet; the first run is the test.`}><Badge kind={e.proven ? "ok" : undefined}>{e.label}{e.proven ? " · proven in a run" : " · not yet run"}</Badge></span>)}
                  </div>
                  {m.notes && <div className="tiny mt" style={{ marginTop: 4 }}>{m.notes}</div>}
                </td>
                <td>{m.suites.join(", ")}</td>
                <td style={{ textAlign: "right" }}>
                  {canEdit && <button className="btn sm danger" disabled={removing === m.id} onClick={() => onRemove(m)}>{removing === m.id ? "Removing…" : "Remove"}</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ------------------------------------------------------------------------- reasoning model

function ReasoningCard({ data, canEdit, onSaved, onError }: { data: SetupData; canEdit: boolean; onSaved: (m: string) => void; onError: (m: string) => void }) {
  const r = data.reasoning;
  const [choice, setChoice] = useState(r.current?.call_id ?? "");
  const [busy, setBusy] = useState(false);
  useEffect(() => setChoice(r.current?.call_id ?? ""), [r.current?.call_id]);
  const save = async () => {
    setBusy(true);
    try { const m = await api.setReasoningModel(choice); onSaved(`${m.name} now writes the hardware proposals and executive summaries.`); }
    catch (e) { onError((e as Error).message); }
    finally { setBusy(false); }
  };
  return (
    <div className="card mb">
      <h2>Reasoning model <Help text="The Amazon Bedrock model that reasons about which GPU and CPU to compare for a model you picked (with live web search) and that writes the executive summary of results. Listed: the Anthropic Claude and OpenAI models this AWS account can call in this Region. Default: the newest Claude Opus." /></h2>
      {r.error && <Note kind="bad">Bedrock could not be asked for its model list: {r.error}</Note>}
      <div className="row">
        <select value={choice} onChange={(e) => setChoice(e.target.value)} disabled={!canEdit || r.models.length === 0} style={{ minWidth: 360 }}>
          {r.models.length === 0 && <option value="">No Claude or OpenAI model is callable here</option>}
          {r.models.map((m) => <option key={m.call_id} value={m.call_id}>{m.name} ({m.provider}) — {m.call_id}</option>)}
        </select>
        {canEdit && <button className="btn primary" disabled={busy || !choice || choice === r.current?.call_id} onClick={save}>{busy ? "Saving…" : "Use this model"}</button>}
        {r.current && <span className="tiny">{r.current.source === "setup" ? "Chosen here." : "Default: the newest Claude Opus the account can call."}</span>}
      </div>
    </div>
  );
}
