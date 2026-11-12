// Typed client for the local API (webui/api/main.py).

export type UseCase = "standard" | "dspm";

export interface ModelInfo {
  id: string; name: string; description: string; kind: string; type: string; params_b: number; license: string;
  commercial_ok: boolean; use_cases: UseCase[]; batch_capable: boolean; batch_sizes_default: number[];
  precisions: Record<string, { label: string; help: string; engines: string[]; quality_only: boolean }>;
}
export interface Hardware {
  id: string; accelerator: "gpu" | "cpu"; software: string[]; engines: string[]; software_help: string;
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
  presets: Record<string, { label: string; help: string; p99_e2e_ms: number }>;
  depths: Record<string, { label: string; help: string; requests: number; passes: number }>;
  request_modes: Record<string, { label: string; help: string }>;
  purchase: Record<string, { label: string; help: string }>;
  concurrency_hints: Record<string, string>;
  precisions: Record<string, { label: string; help: string }>;
  engines: Record<string, { label: string; help: string }>;
  fair_ratio: [number, number]; account: string; region: string;
}

export interface Side { hardware: string; instance_type: string; precisions: string[]; alternates?: string[] }
export interface Wizard {
  use_case: UseCase; prompt_set: string; answer_tokens: number; workloads: string[]; test_items: number; preset: string;
  depth: "quick" | "full" | "custom"; requests: number | null; passes: number | null; models: string[]; precision: string;
  gpu: Side | null; cpus: Side[]; parallel_targets: boolean; request_mode: "single" | "concurrent" | "batch";
  concurrency: number[]; batch_sizes: number[]; sla: Record<string, number | undefined>; purchase: "spot" | "on-demand";
  spot_fallback: boolean; spot_wait_minutes: number; launch_attempts: number; name: string; allow_unfair: boolean;
  sized_for: string | null;          // "<model>|<precision>" the hardware step last pre-selected sizes for
}
export const defaultWizard: Wizard = {
  use_case: "standard", prompt_set: "short", answer_tokens: 128, workloads: ["W1", "W3"], test_items: 300,
  preset: "inline_standard", depth: "quick", requests: null, passes: null, models: [], precision: "bf16",
  gpu: null, cpus: [], parallel_targets: true, request_mode: "concurrent", concurrency: [1, 4, 16], batch_sizes: [8],
  sla: { p99_ttft_ms: 2000, p50_tpot_ms: 150 }, purchase: "spot", spot_fallback: true, spot_wait_minutes: 10, launch_attempts: 30,
  name: "", allow_unfair: false, sized_for: null,
};

export interface Reason { kind: string; ok: boolean; text: string }
export interface Candidate { instance_type: string; spec: InstanceSpec; fit: "recommended" | "possible" | "not_advisable"; reasons: Reason[] }
export interface Proposal { gpu: Candidate[]; cpu: Candidate[]; pairing: Pairing | null; weights_gib: number }
export interface Suggested extends Candidate { ratio: number | null; fair: boolean; role?: "match" | "headroom" | "economy"; role_label?: string; role_help?: string }
export interface Suggestion {
  gpu_type: string; gpu_spec?: InstanceSpec; best: Suggested | null; alternatives: Suggested[]; others: Suggested[]; rejected: Suggested[];
  basis: string[]; weights_gib?: number; error: string | null;
}
export interface GpuSuggestion { best: Suggested | null; alternatives: Suggested[]; others: Suggested[]; rejected: Suggested[]; basis: string[]; weights_gib: number; error: string | null }
export interface InitialSelection { gpu: GpuSuggestion; cpu: Suggestion | null; summary: string | null; gpu_alternates: string[] }
export interface AiStep { kind: "search" | "fetch"; text: string; n?: number; chars?: number; error?: string | null }
export interface AiReview {
  markdown: string; model: { model_id: string; call_id: string; name: string; version: string }; usage: { inputTokens: number; outputTokens: number };
  tool_calls: number; seconds: number; sources: { url: string; title: string }[]; steps: AiStep[]; generated_at: string;
}
export interface AiJob { id: string; status: "running" | "done" | "error"; steps: AiStep[]; tool_calls: number; model: AiReview["model"] | null; seconds: number; error: string | null; result: AiReview | null }
export interface Pairing { cpu: string; gpu: string; ratio: number | null; fair: boolean | null; reason: string; cpu_target?: string; gpu_target?: string }

export interface Plan {
  ok: boolean; problems: string[]; warnings: string[]; grade: "quick" | "full"; request_mode: string; yaml: string;
  series: { model: string; model_id: string; hardware: string; instance_type: string; accelerator: "gpu" | "cpu"; precision: string; engine: string; measurements: number }[];
  pairing: Pairing[];
  estimate: { wall_clock_hours: number; usd: { instances: number; nat_gateway: number; lambda: number; total: number }; note: string } | null;
}

export interface RunRow {
  run_id: string; status: string; status_label: string; updated_at: string; grade: "quick" | "full"; purchase: string; n_cells: number;
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
  run_id: string; status: string; status_label: string; grade: string; purchase: string; updated_at: string; refusal: string[] | null;
  notices: Notice[]; progress: { done: number; total: number }; has_report: boolean; targets: TargetStatus[];
  cells: { model: string; hardware: string; hardware_label: string; precision: string; engine: string; status: string; status_label: string; measurements: number; error: string }[];
  levels: { model: string; hardware: string; hardware_label: string; what: string; conc: number | null; batch: number; pass: number | null; status: string; status_label: string; completed: number; errors: number; detail: string }[];
}
export interface Shape { use_case: UseCase; request_mode: string; concurrency: number[]; grade: string; prompt_set?: string; answer_tokens?: number; workloads?: string[]; dataset?: string; policy?: string; requests_per_measurement?: number; repeats?: number; test_items?: number }
export interface SummaryInfo { run_id: string; shape: Shape; default_sla: Record<string, number | string>; models: string[]; hardware: string[]; instance_types: string[]; smoke: boolean; warnings: string[] }

export interface Accuracy { recall: number; recall_ci: [number, number]; fpr: number; fpr_ci: [number, number]; precision: number; f1: number; format_adherence: number; agreement_with_gpu: number | null; qualifies_vs_gpu: boolean | null; fit: boolean }
export interface Card { series: string; instance_type: string; engine: string; precision: string; conc: number; batch: number; usd_per_hour: number; throughput: number | null; latency: number | null; cost: number | null; quality_delta: number | null; amx: string | null; accuracy: Accuracy | null; run_id: string; ttft_p50: number | null; tpot_p50: number | null; e2e_p50: number | null; e2e_p99: number | null }
export interface Point { series: string; side: "gpu" | "cpu"; conc: number; batch: number; throughput: number | null; latency: number | null; cost: number | null; meets: boolean; reason: string; run_id: string; instance_type: string; precision: string; engine: string; e2e_p50: number | null; e2e_p99: number | null; ttft_p99: number | null; tpot_p50: number | null; req_s: number | null; error_pct: number | null; amx: string | null; quality_delta: number | null; accuracy: Accuracy | null }
export interface Group {
  key: string; label: string; units: { thr: string; thr_label: string; cost: string; cost_label: string; lat: string; lat_label: string; lat_sla: string };
  best_gpu: Card | null; best_cpu: Card | null;
  verdict: { text: string; instances_per_gpu?: number; hourly_cost_ratio?: number; within_limit: boolean | null; limit?: number; missing_side?: "gpu" | "cpu" } | null;
  equivalence: { series: string; instance_type: string; precision: string; conc: number; instances_per_gpu: number; hourly_cost_ratio: number; within_limit: boolean; run_id: string }[];
  points: Point[]; series: string[]; missing: { gpu: string | null; cpu: string | null };
}
export interface NotRun { run_id: string; target: string; side: "GPU" | "CPU"; instance_type: string | null; refusals: number; reason: string }
export interface CompareOut {
  comparability: { ok: boolean; blockers: string[]; warnings: string[]; shapes: Shape[] };
  sla: Record<string, number | string | null>;
  analysis: { groups: Group[]; pairing: Pairing[]; warnings: string[]; smoke: boolean; not_run: NotRun[] } | null;
  runs: { run_id: string; shape: Shape; smoke: boolean }[];
}

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep statusText */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

export const api = {
  catalog: () => j<Catalog>("/api/catalog"),
  data: (uc: UseCase) => j<any>(`/api/data/${uc}`),
  sample: (uc: UseCase, q: Record<string, string | number | undefined>) => {
    const qs = Object.entries(q).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");
    return j<any[]>(`/api/data/${uc}/sample?${qs}`);
  },
  propose: (body: { models: string[]; precision: string; gpu_type?: string | null; cpu_type?: string | null }) =>
    j<Proposal>("/api/propose", { method: "POST", body: JSON.stringify(body) }),
  suggest: (body: { models: string[]; precision: string; gpu_type: string }) =>
    j<Suggestion>("/api/suggest", { method: "POST", body: JSON.stringify(body) }),
  initial: (body: { model: string; precision: string }) =>
    j<InitialSelection>("/api/suggest/initial", { method: "POST", body: JSON.stringify(body) }),
  aiProposeStart: (w: Wizard) => j<{ job_id: string; prompt_preview: string }>("/api/propose/ai", { method: "POST", body: JSON.stringify(w) }),
  aiProposeStatus: (id: string) => j<AiJob>(`/api/propose/ai/${id}`),
  deleteRun: (id: string) => j<{ deleted: string; objects: number; rows: number; status_was: string }>(`/api/runs/${id}`, { method: "DELETE" }),
  plan: (w: Wizard) => j<Plan>("/api/plan", { method: "POST", body: JSON.stringify(w) }),
  start: (w: Wizard) => j<{ run_id: string }>("/api/runs", { method: "POST", body: JSON.stringify(w) }),
  aws: () => j<{ ok: boolean; account?: string; identity?: string; error?: string }>("/api/aws/status"),
  runs: () => j<RunRow[]>("/api/runs"),
  run: (id: string) => j<RunDetail>(`/api/runs/${id}`),
  stop: (id: string) => j<any>(`/api/runs/${id}/stop`, { method: "POST" }),
  summary: (id: string) => j<SummaryInfo>(`/api/runs/${id}/summary`),
  execModel: () => j<{ model_id: string; call_id: string; name: string; version: string }>("/api/executive-summary/model"),
  execSummary: (body: { run_ids: string[]; sla?: Record<string, number | string | null>; limit?: number; models?: string[]; force?: boolean }) =>
    j<ExecSummary>("/api/executive-summary", { method: "POST", body: JSON.stringify(body) }),
  compare: (body: { run_ids: string[]; sla?: Record<string, number | string | null>; limit?: number; models?: string[]; series?: string[] }) =>
    j<CompareOut>("/api/compare", { method: "POST", body: JSON.stringify(body) }),
};

export const fmt = {
  n: (v: number | null | undefined, d = 0) => (v == null || Number.isNaN(v) ? "–" : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })),
  usd: (v: number | null | undefined, d = 2) => (v == null ? "–" : "$" + v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })),
  pct: (v: number | null | undefined) => (v == null ? "–" : (v * 100).toFixed(0) + "%"),
  x: (v: number | null | undefined, d = 1) => (v == null ? "–" : v.toFixed(d) + "x"),
};
