// Typed client for the API (webui/api/main.py). In the cloud deployment every call carries the
// signed-in user's Cognito access token (auth.ts); locally there is no sign-in.
import { accessToken, authEnabled, signIn } from "./auth";

export type UseCase = "standard" | "dspm" | "translation" | "embedding" | "intel";

export interface ModelInfo {
  id: string; name: string; description: string; kind: string; type: string; params_b: number; license: string;
  commercial_ok: boolean; use_cases: UseCase[]; batch_capable: boolean; batch_sizes_default: number[];
  precisions: Record<string, { label: string; help: string; engines: string[]; quality_only: boolean; requires_cpu_flag: string | null }>;
  hf_repo: string; user_added: boolean;
}
export interface Hardware {
  id: string; accelerator: "gpu" | "cpu"; cpu_features: string[]; software: string[]; engines: string[]; software_help: string;
  default_instance_type: string; alternates: string[]; label: string; default_spec: InstanceSpec;
}
export interface InstanceSpec {
  type: string; family: string; vcpus: number; memory_gib: number; accelerator: "gpu" | "cpu"; gpus: number;
  gpu_model: string | null; gpu_memory_gib: number | null; amx: boolean; usd_per_hour: number; current_generation: boolean;
}
export interface Catalog {
  use_cases: Record<UseCase, { name: string; short: string; long: string; icon: string }>;
  models: ModelInfo[]; hardware: Hardware[]; instances: Record<string, InstanceSpec>; price_snapshot: string;
  prompt_sets: Record<string, { label: string; help: string }>;
  workloads: Record<string, { label: string; help: string }>;
  directions: Record<string, { label: string; help: string }>;
  intel_io: Record<string, { label: string; help: string }>;
  text_sizes: Record<string, { label: string; help: string; answer_tokens: number }>;
  presets: Record<string, { label: string; help: string; p99_e2e_ms: number }>;
  depths: Record<string, { label: string; help: string; requests: number; passes: number }>;
  request_modes: Record<string, { label: string; help: string }>;
  purchase: Record<string, { label: string; help: string }>;
  concurrency_hints: Record<string, string>;
  cpu_flag_notes: Record<string, string>;
  precisions: Record<string, { label: string; help: string }>;
  engines: Record<string, { label: string; help: string }>;
  fair_ratio: [number, number]; account: string; region: string;
}

export interface Side { hardware: string; instance_type: string; precisions: string[]; alternates?: string[] }
export interface Wizard {
  use_case: UseCase; prompt_set: string; answer_tokens: number; workloads: string[]; test_items: number; preset: string;
  directions: string[]; text_size: string; translation_items: number; embedding_items: number; intel_io: string;
  depth: "quick" | "full" | "custom"; requests: number | null; passes: number | null; models: string[]; precision: string;
  gpu: Side | null; cpus: Side[]; parallel_targets: boolean; request_mode: "single" | "concurrent" | "batch";
  concurrency: number[]; batch_sizes: number[]; sla: Record<string, number | undefined>; purchase: "spot" | "on-demand";
  spot_fallback: boolean; spot_wait_minutes: number; launch_attempts: number; name: string; allow_unfair: boolean;
  sized_for: string | null;          // "<model>|<precision>" the hardware step last pre-selected sizes for
  copied_from: string | null;        // run id this draft was copied from ("Copy to new run"), for the banner
  recommendation?: string | null;    // what "Re-run with recommended sizes" changed, for the banner
}
export const defaultWizard: Wizard = {
  use_case: "standard", prompt_set: "short", answer_tokens: 128, workloads: ["W1", "W3"], test_items: 300,
  preset: "inline_standard", directions: ["en-de", "de-en", "en-es", "es-en"], text_size: "sentence", translation_items: 100,
  embedding_items: 300, intel_io: "128/128",
  depth: "quick", requests: null, passes: null, models: [], precision: "bf16",
  gpu: null, cpus: [], parallel_targets: true, request_mode: "concurrent", concurrency: [1, 4, 16], batch_sizes: [8],
  sla: { p99_ttft_ms: 2000, p50_tpot_ms: 150 }, purchase: "spot", spot_fallback: true, spot_wait_minutes: 10, launch_attempts: 30,
  name: "", allow_unfair: false, sized_for: null, copied_from: null,
};

// The AI hardware proposal (webui/api/ai_propose.py): reasoned per model with web search, validated against the allowed sizes.
export interface AiStep { kind: "search" | "repair"; text: string; n?: number; error?: string | null }
export interface AiProposal {
  gpu: { instance_type: string; why: string; alternates: string[] };
  cpu: { instance_type: string; why: string };
  summary: string; checked: string[]; caveats: string[]; confidence: "high" | "medium" | "low"; sources: string[];
}
export interface AiResult {
  proposal: AiProposal | null; problem: string | null; markdown: string;
  model: { model_id: string; call_id: string; name: string; provider?: string }; usage: { inputTokens: number; outputTokens: number };
  tool_calls: number; seconds: number; sources: { url: string; title: string }[]; steps: AiStep[]; generated_at: string;
  repaired?: boolean; stop_reason?: string | null;
}
export interface AiJob { id: string; status: "running" | "done" | "error"; steps: AiStep[]; tool_calls: number; model: AiResult["model"] | null; seconds: number; error: string | null; result: AiResult | null }
export interface Pairing { cpu: string; gpu: string; ratio: number | null; fair: boolean | null; reason: string; cpu_target?: string; gpu_target?: string }

export interface Plan {
  ok: boolean; problems: string[]; warnings: string[]; grade: "quick" | "full"; request_mode: string; yaml: string;
  series: { model: string; model_id: string; hardware: string; instance_type: string; accelerator: "gpu" | "cpu"; precision: string; engine: string; measurements: number }[];
  pairing: Pairing[];
  estimate: { wall_clock_hours: number; usd: { instances: number; nat_gateway: number; lambda: number; total: number }; note: string } | null;
}

export interface RunRow {
  run_id: string; name: string | null; model: string | null; status: string; status_label: string; updated_at: string | null;
  grade: "quick" | "full"; purchase: string | null; n_cells: number; errors: string[] | null;
  not_run: { hardware: string; hardware_label: string; reason: string }[];
}
export interface TargetStatus {
  hardware: string; hardware_label: string; instance_id: string | null; instance_type: string | null; state: string | null; state_label: string | null;
  lifecycle: "spot" | "on-demand" | null; requested: string | null; fallback_from_spot: boolean; capacity_refusals: number; interruptions: number;
  attempts_max: number | null; did_not_run: boolean; not_run_reason: string | null;
  launched_at: string | null; status: string; events: { time: string; event: string; instance_id?: string; instance_type?: string; lifecycle?: string; state_reason?: string }[];
}
export interface Notice { kind: "bad" | "warn" | "ok"; title: string; text: string }
export interface ExecSummary { markdown: string; model: { model_id: string; call_id: string; name: string; version: string }; usage: { inputTokens: number; outputTokens: number }; seconds: number; generated_at: string }
export interface RunDetail {
  run_id: string; name: string | null; status: string; status_label: string; grade: string; purchase: string; updated_at: string; refusal: string[] | null;
  failure: string[] | null; notices: Notice[]; progress: { done: number; total: number }; has_report: boolean; targets: TargetStatus[];
  cells: { model: string; hardware: string; hardware_label: string; precision: string; engine: string; status: string; status_label: string; measurements: number; error: string }[];
  levels: { model: string; hardware: string; hardware_label: string; what: string; conc: number | null; batch: number; pass: number | null; status: string; status_label: string; completed: number; errors: number; detail: string }[];
}
export interface Shape { use_case: UseCase; request_mode: string; concurrency: number[]; grade: string; prompt_set?: string; answer_tokens?: number; workloads?: string[]; directions?: string[]; io?: string; dataset?: string; policy?: string; requests_per_measurement?: number; repeats?: number; test_items?: number }
export interface SummaryInfo { run_id: string; shape: Shape; default_sla: Record<string, number | string>; models: string[]; hardware: string[]; instance_types: string[]; smoke: boolean; warnings: string[] }

export interface Accuracy { recall: number; recall_ci: [number, number]; fpr: number; fpr_ci: [number, number]; precision: number; f1: number; format_adherence: number; agreement_with_gpu: number | null; qualifies_vs_gpu: boolean | null; fit: boolean }
export interface EmbeddingQuality { n: number | null; dim: number | null; cosine_mean: number | null; cosine_min: number | null; recall_de: number | null; recall_es: number | null; flag: string | null }
export interface MechanicsSide { instance_type: string; engine: string; precision: string; weights_gb: number | null; bandwidth_gb_s: number | null; peak_tflops: number | null; bandwidth_source: string | null; cores: number | null; low_conc?: number; implied_bandwidth_gb_s?: number; bandwidth_utilization?: number | null; top_conc?: number; achieved_tflops?: number; compute_utilization?: number | null; best_throughput?: number | null }
export interface Mechanics { shape?: { input_tokens: number | null; output_tokens: number | null; writing_share: number | null; generation: boolean }; sides?: Record<"gpu" | "cpu", MechanicsSide>; why?: string; error?: string }
export interface IntelReference { model_id: string; cpu_model: string; sockets: number; total_cores: number; memory_gb: number; ddr5_mt_s: number; io: string; max_concurrency: number; throughput_tok_s: number; tpot_ms: number; ttft_ms: number; vllm_version: string; tested: string | null; tok_s_per_core: number | null }
export interface TranslationQuality { chrf: number | null; chrf_p10: number | null; length_ratio: number | null; truncated_pct: number | null; n: number | null; chrf_vs_gpu: number | null; exact_match_vs_gpu: number | null; flag: string | null }
export interface Saving { basis: string; label: string; usd_per_hour: number; saving_pct: number | null }
export interface Card { series: string; instance_type: string; engine: string; precision: string; conc: number; batch: number; usd_per_hour: number; usd_per_hour_on_demand: number | null; price_label: string | null; savings: Saving[]; throughput: number | null; latency: number | null; cost: number | null; quality_delta: number | null; quality_flag: string | null; amx: string | null; accuracy: Accuracy | null; translation: TranslationQuality | null; embedding: EmbeddingQuality | null; run_id: string; ttft_p50: number | null; tpot_p50: number | null; e2e_p50: number | null; e2e_p99: number | null }
/** How busy the hardware was during one measurement (host-level means of 10-second samples). */
export interface PointUtil { busy_mean: number | null; busy_p95: number | null; busy_max: number | null; mem_max: number | null; gpu_mem_pct: number | null; cpu_mean: number | null; gpu_mean: number | null; cpu_scale?: number }
export interface Point { series: string; side: "gpu" | "cpu"; conc: number; batch: number; throughput: number | null; latency: number | null; cost: number | null; meets: boolean; reason: string; run_id: string; instance_type: string; precision: string; engine: string; e2e_p50: number | null; e2e_p99: number | null; ttft_p99: number | null; tpot_p50: number | null; req_s: number | null; error_pct: number | null; amx: string | null; quality_delta: number | null; quality_flag: string | null; accuracy: Accuracy | null; translation: TranslationQuality | null; embedding: EmbeddingQuality | null; util: PointUtil | null }
/** Was the instance the right size for this load? One verdict per hardware option (bench/utilization.py). */
export interface Suggestion {
  instance_type: string | null; kind: "keep" | "smaller" | "bigger" | "stronger" | "growth" | "none"; why: string;
  usd_per_hour: number | null; usd_delta_per_hour: number | null; pct_of_current?: number; vcpus?: number; memory_gib?: number; gpus?: number;
  gpu_model?: string | null; gpu_memory_gib?: number | null; pairing?: { other: string; ratio: number | null; fair: boolean | null; text: string };
}
export interface Sizing {
  series: string; side: "gpu" | "cpu"; instance_type: string | null; engine: string | null; precision: string | null; size?: string; run_id?: string;
  suggestion?: Suggestion | null;
  verdict: "over" | "good" | "limit" | "under" | "unknown"; label: string; text: string; notes?: string[]; busy_basis?: string;
  util_top?: number; util_p95_top?: number | null; util_low?: number; load_top?: string; meets_top?: boolean; measurements: number; with_samples: number;
  points: { conc: number; batch: number; meets: boolean; util_mean: number | null; util_p95: number | null; mem_pct: number | null }[];
}
export interface UtilSeries { t: number[]; period_s: number | null; cpu?: (number | null)[]; mem?: (number | null)[]; gpu?: (number | null)[]; gpu_mem?: (number | null)[]; gpu_power?: (number | null)[] }
export interface UtilTarget {
  target: string; instance_type: string | null; accelerator: "gpu" | "cpu"; spec: InstanceSpec | null; series: UtilSeries | null; hosts: number;
  scoped_to_run: boolean; instances: number;
  gpu_mem_total_mib: number | null; windows: { level: string; name: string | null; conc: number; batch: number; start: number; end: number }[]; error: string | null;
}
export interface TechEngine { engine: string; label: string; help: string; image: string; built: boolean; dockerfile: string | null; port: number; env: Record<string, string>; args: Record<string, unknown> }
export interface TechHardware { id: string; label: string; accelerator: "gpu" | "cpu"; instance_type: string; alternates: string[]; cpu_features: string[]; engines: TechEngine[] }
export interface TechImage { name: string; dockerfile: string; base_image: string; runs_on: string; purpose: string }
export interface TechModelFlags { model: string; name: string; hf_repo: string; engine: string; args: string[]; notes: string | null }
export interface Technical {
  hardware: TechHardware[]; images: TechImage[]; model_flags: TechModelFlags[]; files: Record<string, { title: string; text: string | null; present: boolean }>;
  thin_tasks: Record<string, string>; export_presets: Record<string, Record<string, unknown>>; precisions: Record<string, { label: string; help: string }>;
  diagram: { svg: string; drawio: string; present: boolean }; harness_commit: string | null;
}
export interface RunUtilization { run_id: string; start?: number; end?: number; period_s?: number; targets: UtilTarget[]; retention_note?: string; reason?: string }
export interface Group {
  key: string; label: string; units: { thr: string; thr_label: string; cost: string; cost_label: string; lat: string; lat_label: string; lat_sla: string };
  best_gpu: Card | null; best_cpu: Card | null;
  verdict: { text: string; instances_per_gpu?: number; hourly_cost_ratio?: number; within_limit: boolean | null; limit?: number; missing_side?: "gpu" | "cpu" } | null;
  sizing: Sizing[];
  equivalence: { series: string; instance_type: string; precision: string; conc: number; instances_per_gpu: number; hourly_cost_ratio: number; within_limit: boolean; run_id: string }[];
  points: Point[]; series: string[]; missing: { gpu: string | null; cpu: string | null };
  mechanics?: Mechanics;
  demand_plan?: DemandPlan | { error: string } | null;
}
export interface DemandOption { series: string; side: "gpu" | "cpu"; instance_type: string; engine: string; precision: string; capacity_req_s: number; capacity_basis: string; usd_per_hour_each: number; price_label: string; estimated: boolean; instances: number; usd_per_hour: number; usd_per_month: number; headroom: number | null; capacity_at?: { conc: number; batch: number; busy_mean: number | null; latency_p99_ms: number | null } }
export interface DemandPlan { demand_req_s: number; demand_rpm: number; options: DemandOption[]; cheapest: DemandOption | null; cheapest_measured: DemandOption | null; cheapest_gpu: DemandOption | null; cheapest_cpu: DemandOption | null; cheapest_cpu_measured: DemandOption | null; crossover: { req_s: number | null; rpm: number | null; kind: string; text: string } | null; curve: { rpm: number; gpu_usd_per_hour: number | null; cpu_usd_per_hour: number | null; cpu_measured_usd_per_hour: number | null }[]; text: string }
export interface Pricing { basis: string; cpu_usd_per_hour?: number | null; gpu_usd_per_hour?: number | null }
export interface TypicalLoad { key: string; label: string; requests_per_minute: number; help: string }
export interface NotRun { run_id: string; target: string; side: "GPU" | "CPU"; instance_type: string | null; refusals: number; reason: string }
export interface CompareOut {
  comparability: { ok: boolean; blockers: string[]; warnings: string[]; shapes: Shape[] };
  sla: Record<string, number | string | null>;
  analysis: { groups: Group[]; pairing: Pairing[]; warnings: string[]; smoke: boolean; not_run: NotRun[]; intel_reference?: IntelReference[];
    pricing?: { basis: string; label: string; cpu_usd_per_hour: number | null; gpu_usd_per_hour: number | null; snapshot: string | null };
    demand_rpm?: number | null; typical_loads?: TypicalLoad[]; how_to_read?: string } | null;
  runs: { run_id: string; shape: Shape; smoke: boolean }[];
}

/** An HTTP failure with its status, so callers can tell "signed out" (401) from "service restarting" (5xx) from "no network" (status 0). */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) { super(message); this.status = status; }
}
async function authed(url: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {};
  const token = await accessToken();                       // null locally (no sign-in)
  if (token) headers.Authorization = `Bearer ${token}`;
  let r: Response;
  try { r = await fetch(url, { ...init, headers: { ...headers, ...(init?.headers as Record<string, string> | undefined) } }); }
  catch (e) { throw new ApiError(`The service did not answer (${(e as Error).message}).`, 0); }
  if (r.status === 401 && authEnabled()) {
    // The session is gone (expired refresh token, signed out elsewhere): back through the hosted login.
    void signIn();
    throw new ApiError("Your session has ended; taking you to the sign-in page.", 401);
  }
  return r;
}
/** A file behind the API (the architecture diagram), fetched with the sign-in token: an <img src> or a plain link cannot carry it. */
export async function fetchBlob(url: string): Promise<Blob> {
  const r = await authed(url);
  if (!r.ok) throw new ApiError(`HTTP ${r.status}`, r.status);
  return r.blob();
}
async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await authed(url, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers as Record<string, string> | undefined) } });
  if (!r.ok) {
    // CloudFront forwards a 500 with an empty statusText and a non-JSON body; an empty message read as
    // "no error" downstream (the Estimate panel spun forever), so always name the status.
    let msg: unknown = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep statusText */ }
    const text = typeof msg === "string" ? msg : JSON.stringify(msg);
    throw new ApiError(text && text !== "undefined" ? `${text} (HTTP ${r.status})` : `The server answered HTTP ${r.status} without a reason; check the API log.`, r.status);
  }
  return r.json();
}

export interface AwsStatus { ok: boolean; account?: string; identity?: string; error?: string; expired?: boolean; credentials?: string }

// Setup page (webui/api/settings.py, onboarding.py, catalog_store.py)
export interface HfTokenStatus { configured: boolean; available: boolean; name?: string; last_changed?: string | null; reason?: string }
export interface ReasoningModel { model_id: string; call_id: string; name: string; provider: string; family: string; source?: "setup" | "default" }
export interface SetupModel {
  id: string; name: string; repo: string; revision: string; type: string; kind: string; params_b: number; license: string; gated: boolean;
  suites: string[]; user_added: boolean; enabled: boolean; notes: string | null;
  engines: { id: string; label: string; proven: boolean }[];
}
export interface OnboardJob {
  id: string; repo: string; status: "running" | "done" | "failed"; stage: string; elapsed_s: number; notes: string[]; error: string | null;
  model: { id: string; name: string; type: string; params_b: number; license: string; gated: boolean } | null;
  pools?: { prompt_files: number; dspm_files: number };
}
export interface Setup {
  hf_token: HfTokenStatus; models: SetupModel[]; catalog_source: string | null; proven_error: string | null;
  reasoning: { models: ReasoningModel[]; current: ReasoningModel | null; error?: string }; running_job: OnboardJob | null;
}

export const api = {
  setup: () => j<Setup>("/api/setup"),
  setHfToken: (token: string) => j<HfTokenStatus>("/api/setup/hf-token", { method: "POST", body: JSON.stringify({ token }) }),
  setReasoningModel: (call_id: string) => j<ReasoningModel>("/api/setup/reasoning-model", { method: "POST", body: JSON.stringify({ call_id }) }),
  addModel: (repo: string) => j<{ job_id: string }>("/api/setup/models", { method: "POST", body: JSON.stringify({ repo }) }),
  addModelJob: (id: string) => j<OnboardJob>(`/api/setup/models/jobs/${id}`),
  removeModel: (id: string) => j<{ removed: string; remaining: number }>(`/api/setup/models/${encodeURIComponent(id)}`, { method: "DELETE" }),
  catalog: () => j<Catalog>("/api/catalog"),
  data: (uc: UseCase) => j<any>(`/api/data/${uc}`),
  sample: (uc: UseCase, q: Record<string, string | number | undefined>) => {
    const qs = Object.entries(q).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");
    return j<any[]>(`/api/data/${uc}/sample?${qs}`);
  },
  aiProposeStart: (w: Wizard) => j<{ job_id: string; prompt_preview: string }>("/api/propose/ai", { method: "POST", body: JSON.stringify(w) }),
  aiProposeStatus: (id: string) => j<AiJob>(`/api/propose/ai/${id}`),
  wizardFrom: (id: string) => j<Wizard>(`/api/runs/${id}/wizard`),
  wizardRecommended: (id: string) => j<Wizard & { changes: string[] }>(`/api/runs/${id}/wizard/recommended`),
  technical: () => j<Technical>("/api/technical"),
  deleteRun: (id: string) => j<{ deleted: string; objects: number; rows: number; status_was: string; terminated_instances: string[] }>(`/api/runs/${id}`, { method: "DELETE" }),
  slug: (name: string, use_case: string) => j<{ slug: string; run_id: string }>(`/api/runs/slug?name=${encodeURIComponent(name)}&use_case=${encodeURIComponent(use_case)}`),
  plan: (w: Wizard) => j<Plan>("/api/plan", { method: "POST", body: JSON.stringify(w) }),
  start: (w: Wizard) => j<{ run_id: string }>("/api/runs", { method: "POST", body: JSON.stringify(w) }),
  aws: () => j<AwsStatus>("/api/aws/status"),
  runs: () => j<RunRow[]>("/api/runs"),
  run: (id: string) => j<RunDetail>(`/api/runs/${id}`),
  stop: (id: string) => j<any>(`/api/runs/${id}/stop`, { method: "POST" }),
  summary: (id: string) => j<SummaryInfo>(`/api/runs/${id}/summary`),
  utilization: (id: string) => j<RunUtilization>(`/api/runs/${id}/utilization`),
  /** The Markdown report, fetched with the session token and saved as a file (a plain link cannot carry the token). */
  downloadReport: async (id: string) => {
    const token = await accessToken();
    const r = await fetch(`/api/runs/${id}/report.md`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
    if (r.status === 401 && authEnabled()) { void signIn(); throw new Error("Your session has ended; signing you in again."); }
    if (!r.ok) throw new Error(`report: HTTP ${r.status}`);
    const url = URL.createObjectURL(await r.blob());
    const a = Object.assign(document.createElement("a"), { href: url, download: `${id}-report.md` });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
  execModel: () => j<{ model_id: string; call_id: string; name: string; version: string }>("/api/executive-summary/model"),
  execSummary: (body: { run_ids: string[]; sla?: Record<string, number | string | null>; limit?: number; models?: string[]; force?: boolean; pricing?: Pricing; demand_rpm?: number | null }) =>
    j<ExecSummary>("/api/executive-summary", { method: "POST", body: JSON.stringify(body) }),
  compare: (body: { run_ids: string[]; sla?: Record<string, number | string | null>; limit?: number; models?: string[]; series?: string[]; pricing?: Pricing; demand_rpm?: number | null }) =>
    j<CompareOut>("/api/compare", { method: "POST", body: JSON.stringify(body) }),
};

export const fmt = {
  n: (v: number | null | undefined, d = 0) => (v == null || Number.isNaN(v) ? "–" : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })),
  usd: (v: number | null | undefined, d = 2) => (v == null ? "–" : "$" + v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })),
  pct: (v: number | null | undefined) => (v == null ? "–" : (v * 100).toFixed(0) + "%"),
  x: (v: number | null | undefined, d = 1) => (v == null ? "–" : v.toFixed(d) + "x"),
};
