// Technical details: how the application is built and how a benchmark run works, for the engineer who
// wants to check the method and for the reader who is not one. The architecture diagram
// (docs/architecture.drawio, exported to SVG) is fetched with the sign-in token (an <img src> cannot
// carry it, so a plain URL answered 401). Every table column carries a plain-language help icon; the
// dense engine-specific facts sit behind "Technical specifics" so the page reads top-down.
import { useEffect, useState } from "react";
import { api, fetchBlob, type TechHardware, type Technical as TechData } from "../api";
import { Badge, Help, Note, Spinner } from "../components/ui";

export default function Technical() {
  const [data, setData] = useState<TechData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.technical().then(setData).catch((e) => setErr(e.message)); }, []);
  return (
    <div>
      <div className="topbar"><div><h1>Technical details</h1><div className="muted">What runs where, and what happens to a model before it is served. Everything here is read from the deployed build{data?.harness_commit ? ` (commit ${data.harness_commit.slice(0, 10)})` : ""}.</div></div></div>
      {err && <Note kind="bad">{err}</Note>}
      {!data && !err && <Spinner label="Loading…" />}
      {data && (
        <>
          <Architecture data={data} />
          <RunFlow />
          <Hardware hardware={data.hardware} />
          <ModelFlags data={data} />
          <Images data={data} />
          <Preparation data={data} />
          <Files data={data} />
        </>
      )}
    </div>
  );
}

/** A container image reference, readable: registry/name:tag and the first 12 hex digits of the digest; the full reference on hover. */
function ImageRef({ image }: { image: string }) {
  const m = image.match(/^(.*?)(@sha256:([0-9a-f]+))?$/);
  const name = m?.[1] ?? image; const digest = m?.[3];
  return <span title={image} style={{ overflowWrap: "anywhere" }}>{name}{digest ? <span className="muted"> @sha256:{digest.slice(0, 12)}…</span> : null}</span>;
}

/** Column header with a plain-language explanation of what the column means and how to read its values. */
function Th({ children, help, className }: { children: React.ReactNode; help: string; className?: string }) {
  return <th className={className}>{children}<Help text={help} /></th>;
}

function Architecture({ data }: { data: TechData }) {
  const [svgUrl, setSvgUrl] = useState<string | null>(null);
  const [svgErr, setSvgErr] = useState<string | null>(null);
  useEffect(() => {
    if (!data.diagram.present) return;
    let url: string | null = null;
    fetchBlob(data.diagram.svg).then((b) => { url = URL.createObjectURL(b); setSvgUrl(url); }).catch((e) => setSvgErr((e as Error).message));
    return () => { if (url) URL.revokeObjectURL(url); };
  }, [data.diagram.present, data.diagram.svg]);
  const download = async (path: string, name: string) => {
    try {
      const b = await fetchBlob(path);
      const u = URL.createObjectURL(b);
      const a = document.createElement("a"); a.href = u; a.download = name; a.click();
      setTimeout(() => URL.revokeObjectURL(u), 10000);
    } catch (e) { setSvgErr((e as Error).message); }
  };
  return (
    <div className="card mb">
      <div className="row between">
        <h2 style={{ margin: 0 }}>Architecture <Help text="A picture of the AWS pieces the benchmark is made of and how they talk to each other. Drawn with draw.io (docs/architecture.drawio in the repository); download the .drawio file to edit it." /></h2>
        {data.diagram.present && (
          <div className="row">
            <button className="btn sm" disabled={!svgUrl} onClick={() => svgUrl && window.open(svgUrl, "_blank", "noopener")} title="Open the picture alone in a new tab, at full size">Open full size</button>
            <button className="btn sm" onClick={() => download(data.diagram.drawio, "architecture.drawio")} title="The editable source of this picture, for draw.io">Download .drawio</button>
          </div>
        )}
      </div>
      {data.diagram.present ? (
        <div className="mt" style={{ background: "#fff", borderRadius: 12, border: "1px solid var(--line)", padding: 8, overflow: "auto", minHeight: 120 }}>
          {svgUrl && <img src={svgUrl} alt="Architecture diagram" style={{ width: "100%", height: "auto", display: "block" }} />}
          {!svgUrl && !svgErr && <div style={{ color: "#555", padding: 20 }}><Spinner label="Loading the diagram…" /></div>}
          {svgErr && <div style={{ color: "#a00", padding: 20 }}>The diagram could not be loaded: {svgErr}</div>}
        </div>
      ) : <Note kind="warn">The diagram was not included in this build. Run <code>tools/architecture_diagram.py</code> and redeploy.</Note>}
      <div className="small mt">
        <b>Reading it, left to right.</b>
        <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
          <li><b>You</b> reach the <b>web application</b> through CloudFront; it is the only public entry point, and every call needs your sign-in. Only members of the <i>operators</i> group may start runs.</li>
          <li>A run is a <b>Step Functions</b> workflow. Its Lambda functions start <b>one EC2 test machine per hardware option</b> inside a private network, put the model on it, start the serving software in a container and send the test requests to it from inside the network.</li>
          <li>Every raw measurement goes to <b>S3</b>; the <b>DynamoDB</b> table tracks what is running; <b>CloudWatch</b> keeps each machine's busy-ness; <b>Bedrock</b> (Claude) writes the hardware proposal and the executive summary.</li>
          <li>No test machine has a public address. They fetch models and container images through a NAT gateway.</li>
        </ul>
      </div>
    </div>
  );
}

const STEPS: [string, string, string][] = [
  ["Plan", "Your choices in the wizard become a run definition. A pre-check refuses anything that cannot work and says why.",
    "The run definition is run.yaml. The pre-check Lambda resolves every cell (model × hardware option × number format), checks the request pools, the instance types' availability and price, and the Hugging Face token for gated models."],
  ["Launch", "One EC2 machine is started per hardware option: Spot first, On-Demand after the waiting time you set.",
    "target_control launches from the option's launch template, tagged bench:managed. At boot the host syncs the host scripts from s3://…/artifacts/host/, starts the CloudWatch agent and uploads its facts (CPU flags, kernel, driver)."],
  ["Stage the model", "The exact model version is copied onto the machine, from our S3 cache when we have it, otherwise from Hugging Face.",
    "stage_model.sh puts the pinned revision under /opt/models; every LFS file is checked against the sha256 the Hub reports, then cached in S3 for the next run. The token is read from Secrets Manager into a 0600 tmpfs file only for gated repositories."],
  ["Convert, when the engine needs it", "Nothing is converted any more: every active option serves the published model files through vLLM. (The OpenVINO options, which needed a one-off conversion, were retired on 2026-11-21; earlier runs still show their results.)",
    "Until 2026-11-21 an export container turned the checkpoint into OpenVINO IR (BF16-OV, INT8, INT4) and cached it in S3 under a hash of the arguments, revision and tool versions. The code stays in servers/export for reference; no active target uses it."],
  ["Serve", "The serving software starts in a container with the model; the run waits until it answers a real request.",
    "serve.sh runs the engine image with the cell's options (port, environment, model directory, GPU access), waits for the health endpoint, then sends one request; the run page shows the server's answer when that fails."],
  ["Probe", "On CPU machines we check that the matrix math really ran on AMX, and measure the machine's memory speed.",
    "The AMX probe captures a oneDNN / OpenVINO trace of one request and perf counters (exe.amx_busy) during the measurements; a CPU result is AMX VERIFIED only when the GEMMs ran on AMX. A STREAM probe records the host's memory bandwidth."],
  ["Measure", "For every load setting, warm up, then send the test requests and record how long each took. Repeats are pooled.",
    "The level state machine warms the server, then worker Lambdas send the pool's requests at the set concurrency or batch size, recording per-request timings and token counts. aggregate_level summarizes each measurement and reads the host's utilization from CloudWatch; aggregate_run writes the report."],
  ["Tear down", "Machines are terminated when their part is done. The report and every raw record stay in S3.",
    "The reaper terminates anything left behind; artifacts live under runs/<id>/."],
];

function RunFlow() {
  return (
    <div className="card mb">
      <h2>How a benchmark runs <Help text="The eight steps every run goes through, in order. The first sentence of each is the plain version; 'Technical specifics' opens the exact component names for the engineer." /></h2>
      <ol className="small" style={{ paddingLeft: 20, margin: "6px 0 0" }}>
        {STEPS.map(([t, plain, tech]) => (
          <li key={t} style={{ marginBottom: 8 }}>
            <b>{t}.</b> {plain}
            <details className="tiny" style={{ marginTop: 2 }}><summary className="muted" style={{ cursor: "pointer" }}>Technical specifics</summary><div style={{ marginTop: 4 }}>{tech}</div></details>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Hardware({ hardware }: { hardware: TechHardware[] }) {
  return (
    <div className="card mb">
      <h2>Hardware options and their containers <Help text="Each hardware option you can tick in the wizard is a machine type plus the serving software that runs on it. This table shows exactly which software (container image) runs where, so a result can be reproduced." /></h2>
      <div className="small muted">Every hardware option is a Docker container on an EC2 machine: the same model files, a different serving stack. Public images are pinned to an exact version (digest) so a run is reproducible; the ones this project builds itself are listed in the next section.</div>
      <table className="tbl mt">
        <thead><tr>
          <Th help="The name you see in the wizard. GPU = NVIDIA graphics card; CPU = Intel Xeon with AMX (its built-in matrix-math unit). 'needs …' lists a CPU feature the machine must have.">Hardware option</Th>
          <Th help="The AWS machine type used by default, and the alternates tried when it is not available in the chosen Availability Zone.">Instance</Th>
          <Th help="The serving software that loads the model and answers requests. Every active option uses vLLM, on the GPU and on the AMX CPU (Intel's own Xeon recipe). The small PyTorch server this project wrote exists on the GPU option only for models vLLM cannot serve. The PyTorch and OpenVINO CPU options were retired (2026-11-20/21): they measured 3-10x and 2.3x below vLLM on the same instance, comparing engines rather than CPU with GPU.">Engine</Th>
          <Th help="The Docker image that is run. 'Public, pinned' = a vendor image fetched from a public registry at an exact digest, so the same bytes run every time. 'Built by this project' = built from a Dockerfile in this repository when the stack is deployed and stored in this account's registry.">Container image</Th>
          <Th help="The port the server listens on, plus the environment variables and command-line arguments the host passes to the container. These are the tuning settings; they are the same for every run.">How it is started</Th>
        </tr></thead>
        <tbody>
          {hardware.flatMap((h) => h.engines.map((e, i) => (
            <tr key={h.id + e.engine}>
              {i === 0 && <td rowSpan={h.engines.length}><Badge kind={h.accelerator}>{h.accelerator.toUpperCase()}</Badge> <b>{h.label}</b><div className="tiny">{h.id}{h.cpu_features.length ? ` · needs ${h.cpu_features.join(", ")}` : ""}</div></td>}
              {i === 0 && <td rowSpan={h.engines.length}>{h.instance_type}{h.alternates.length ? <div className="tiny">alternates: {h.alternates.join(", ")}</div> : null}</td>}
              <td><b>{e.label}</b><div className="tiny">{e.engine}</div></td>
              <td className="tiny" style={{ minWidth: 240 }}>
                {e.built
                  ? <><Badge kind="accent">Built by this project</Badge><div style={{ marginTop: 4 }}>image <b>{e.image.slice(6)}</b> from <code>{e.dockerfile}</code> (see the next table)</div></>
                  : <><Badge>Public, pinned</Badge><div style={{ marginTop: 4 }}><ImageRef image={e.image} /></div></>}
              </td>
              <td className="tiny">
                port {e.port}
                {Object.keys(e.env).length > 0 && <div>env: {Object.entries(e.env).map(([k, v]) => `${k}=${v}`).join(" · ")}</div>}
                {Object.keys(e.args).length > 0 && <div>args: {Object.entries(e.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" · ")}</div>}
              </td>
            </tr>
          )))}
        </tbody>
      </table>
    </div>
  );
}

function ModelFlags({ data }: { data: TechData }) {
  if (!data.model_flags?.length) return null;
  return (
    <div className="card mb">
      <h2>Model-specific serving flags <Help text="A few models need extra start-up options for vLLM, taken from the vendor's published recipe. They are added on both the GPU and the CPU vLLM option, so neither side gets a setting the other lacks." /></h2>
      <div className="small muted">Most models run with the engine's defaults. These need extra <code>vllm serve</code> flags; the harness adds them on both the GPU and the CPU vLLM option.</div>
      {data.model_flags.map((m) => (
        <div key={m.model} className="mt">
          <b>{m.name}</b> <span className="tiny muted">({m.hf_repo}, {m.engine})</span>
          <pre style={{ marginTop: 4 }}>{"vllm serve <model> ... " + m.args.map((a) => (a.includes(" ") || a.includes("{") ? `'${a}'` : a)).join(" ")}</pre>
          {m.notes && <div className="small muted">{m.notes}</div>}
        </div>
      ))}
    </div>
  );
}

function Images({ data }: { data: TechData }) {
  return (
    <div className="card mb">
      <h2>Images this project builds <Help text="Container images built from this repository when the stack is deployed (the vendor images in the table above are not built here). Each is built from a Dockerfile on top of a public base image." /></h2>
      <table className="tbl mt">
        <thead><tr>
          <Th help="The image's name in config/images.yaml; the hardware table above refers to it as 'image <name>'.">Image</Th>
          <Th help="The recipe the image is built from (a Dockerfile in the repository). Open it verbatim in 'The files' section at the bottom of this page.">Dockerfile</Th>
          <Th help="The public image the recipe starts from (for example a PyTorch or OpenVINO release), pinned to an exact version.">Base image</Th>
          <Th help="Which machines run this image, and for which hardware option.">Runs on</Th>
          <Th help="Why the image exists: what it does that a vendor image does not.">What it is for</Th>
        </tr></thead>
        <tbody>
          {data.images.map((im) => (
            <tr key={im.name}><td><b>{im.name}</b></td><td className="tiny">{im.dockerfile}</td><td className="tiny" style={{ minWidth: 220 }}>{im.base_image.startsWith("(") ? im.base_image : <ImageRef image={im.base_image} />}</td><td className="small">{im.runs_on}</td><td className="small">{im.purpose}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One engine's preparation path: the plain story as bullets, the dense facts behind a toggle. */
function Path({ title, conversion, bullets, specifics }: { title: string; conversion: string; bullets: React.ReactNode[]; specifics: React.ReactNode }) {
  return (
    <div className="tile" style={{ cursor: "default" }}>
      <div className="t">{title}</div>
      <div className="small"><Badge kind={conversion === "No conversion" ? "ok" : "accent"}>{conversion}</Badge></div>
      <ul className="small" style={{ margin: "8px 0 0", paddingLeft: 18 }}>{bullets.map((b, i) => <li key={i} style={{ marginBottom: 4 }}>{b}</li>)}</ul>
      <details className="tiny mt"><summary className="muted" style={{ cursor: "pointer" }}>Technical specifics</summary><div style={{ marginTop: 6, lineHeight: 1.5 }}>{specifics}</div></details>
    </div>
  );
}

function Preparation({ data }: { data: TechData }) {
  return (
    <div className="card mb">
      <h2>What happens to a model before each engine can serve it <Help text="Models are published on Hugging Face in PyTorch's file format. vLLM, the engine on every active option, reads that directly: nothing is converted, so what is measured is the published model. The retired OpenVINO path, which did convert, is described for the earlier runs that used it." /></h2>
      <div className="small muted">Why this matters: a conversion is a step between the published model and what is measured. Every active option runs the published files as they are.</div>
      <div className="grid c2 mt">
        <Path title="vLLM (GPU and AMX CPU) · BF16, W8A8" conversion="No conversion"
          bullets={[
            "The published model files are mounted read-only into the container and loaded as they are.",
            "On the GPU the model runs in BF16. On the CPU, vLLM's CPU backend uses one thread per physical core and runs the matrix math on AMX.",
            "W8A8 (8-bit) cells use a pre-quantized model published on Hugging Face, listed on the model; nothing is quantized here.",
          ]}
          specifics={<>The pinned safetensors checkpoint is mounted at <code>/models</code>. CPU: <code>VLLM_CPU_OMP_THREADS_BIND=auto</code>, <code>VLLM_CPU_NUM_OF_RESERVED_CPU=1</code>; oneDNN executes the GEMMs with AMX ukernels (the AMX probe counts them). The KV-cache budget is sized from the host's memory minus the engine's measured footprint (bench/cellspec.py).</>} />
        <Path title="PyTorch (thin server, GPU only) · BF16" conversion="No conversion"
          bullets={[
            "A small server written for this project loads the published model with the Transformers library and answers an OpenAI-style API; it runs on the GPU option only, for models vLLM cannot serve.",
            "It picks the right loader from the model's kind (chat, classifier, detector, embedding).",
            "The CPU PyTorch option was retired on 2026-11-20: 3-10x below vLLM on the same instance.",
          ]}
          specifics={<><code>servers/thin</code> uses <code>AutoModelFor*</code> chosen from <code>THIN_TASK</code>: {Object.entries(data.thin_tasks).map(([k, v]) => `${k} → ${v}`).join(", ")}. CPU: <code>OMP_NUM_THREADS</code> = physical cores with cores pinned.</>} />
        <div className="tile" style={{ cursor: "default", opacity: 0.75 }}>
          <div className="t">OpenVINO (Model Server and thin server) · retired 2026-11-21</div>
          <div className="small"><Badge kind="warn">No longer offered</Badge></div>
          <ul className="small" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
            <li>OpenVINO needed the model converted into its own format (IR) first, with INT8/INT4 weight compression applied at that step.</li>
            <li>Measured at full grade on RoBERTa: OpenVINO BF16 ran 2.3x below vLLM BF16 on the same m8i.4xlarge, and the weight-only INT8 export was 18% slower than BF16 (an encoder is compute-bound; decompressing weights costs more than the saved bandwidth).</li>
            <li>It therefore compared engines, not CPU with GPU, and was retired together with Flan-T5, whose only CPU engines were PyTorch and OpenVINO. Earlier runs that used it still render.</li>
          </ul>
        </div>
      </div>
      <Note kind="warn"><b>A model added under Setup follows the same paths by its kind.</b> Nothing checks in advance that the engine supports the model family; the first run is the proof, and the run page quotes the server's answer when a container refuses the model.</Note>
    </div>
  );
}

function Files({ data }: { data: TechData }) {
  const order = ["servers/thin/Dockerfile.torch", "servers/thin/Dockerfile.openvino", "servers/export/Dockerfile", "webui/Dockerfile", "host/stage_model.sh", "host/serve.sh", "servers/export/export.py", "servers/export/optimum_export.py"];
  return (
    <div className="card mb">
      <h2>The files, verbatim <Help text="The build recipes and host scripts exactly as they are in this deployed build, for anyone who wants to check the method line by line. Click a file name to open it." /></h2>
      {order.filter((k) => data.files[k]).map((k) => {
        const f = data.files[k];
        return (
          <details key={k} className="mt">
            <summary className="small"><b>{k}</b> — {f.title}{f.present ? "" : " (not in this build)"}</summary>
            {f.text && <pre className="small" style={{ background: "var(--bg-3)", border: "1px solid var(--line)", borderRadius: 10, padding: 12, overflow: "auto", maxHeight: 520, fontSize: 11.5, lineHeight: 1.45 }}>{f.text}</pre>}
          </details>
        );
      })}
    </div>
  );
}
