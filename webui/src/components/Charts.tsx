import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, type Group, type Point, type UtilTarget } from "../api";

// One hue per software stack, warm for GPU and cool for CPU, so the same software always has the
// same colour on every chart and no two lines on a chart share a shade (three blues were
// indistinguishable). Series that still collide (same software, another format or size) get a
// different dash pattern and marker.
const GPU_HUES: Record<string, string> = { vLLM: "#f28c28", PyTorch: "#e04f5f", OpenVINO: "#d9a400" };
const GPU_SPARE = ["#ff6f91", "#c86b2b", "#b5179e"];
const CPU_HUES: Record<string, string> = { vLLM: "#3b82f6", OpenVINO: "#0f9f8f", PyTorch: "#9b5de5" };
const CPU_SPARE = ["#d6409f", "#7cb518", "#0ea5e9", "#8d6e63"];
const DASHES = [undefined, "7 4", "2 4", "10 3 2 3"];
const SHAPES = ["circle", "square", "diamond", "triangle"] as const;

export interface SeriesStyle { color: string; dash?: string; shape: (typeof SHAPES)[number] }

function software(series: string): string {
  const parts = series.split(" · ");
  return parts.length >= 3 ? parts[2].replace(/ ·b\d+$/, "") : "";
}

export function seriesStyles(series: string[]): Record<string, SeriesStyle> {
  const out: Record<string, SeriesStyle> = {};
  const used: Record<string, number> = {};
  let gs = 0, cs = 0;
  for (const s of [...series].sort()) {
    const gpu = s.startsWith("GPU");
    const sw = software(s);
    let color = gpu ? GPU_HUES[sw] : CPU_HUES[sw];
    if (!color) color = gpu ? GPU_SPARE[gs++ % GPU_SPARE.length] : CPU_SPARE[cs++ % CPU_SPARE.length];
    const n = used[color] ?? 0;
    used[color] = n + 1;
    out[s] = { color, dash: DASHES[n % DASHES.length], shape: SHAPES[n % SHAPES.length] };
  }
  return out;
}

export function seriesColors(series: string[]): Record<string, string> {
  return Object.fromEntries(Object.entries(seriesStyles(series)).map(([k, v]) => [k, v.color]));
}

export function Swatch({ style }: { style: SeriesStyle }) {
  return (
    <svg width="26" height="12" style={{ verticalAlign: "middle", marginRight: 6, flex: "none" }} aria-hidden>
      <line x1="1" y1="6" x2="25" y2="6" stroke={style.color} strokeWidth="2.5" strokeDasharray={style.dash} />
      <Marker cx={13} cy={6} r={4} style={style} />
    </svg>
  );
}

function Marker({ cx, cy, r, style }: { cx: number; cy: number; r: number; style: SeriesStyle }) {
  const p = { fill: "var(--bg-2)", stroke: style.color, strokeWidth: 2 };
  if (style.shape === "square") return <rect x={cx - r} y={cy - r} width={2 * r} height={2 * r} {...p} />;
  if (style.shape === "diamond") return <polygon points={`${cx},${cy - r - 1} ${cx + r + 1},${cy} ${cx},${cy + r + 1} ${cx - r - 1},${cy}`} {...p} />;
  if (style.shape === "triangle") return <polygon points={`${cx},${cy - r - 1} ${cx + r + 1},${cy + r} ${cx - r - 1},${cy + r}`} {...p} />;
  return <circle cx={cx} cy={cy} r={r} {...p} />;
}

function toRows(points: Point[], field: keyof Point) {
  const xs = Array.from(new Set(points.map((p) => p.conc))).sort((a, b) => a - b);
  return xs.map((x) => {
    const row: Record<string, number | string | null> = { conc: x };
    for (const p of points) if (p.conc === x) row[p.series + (p.batch > 1 ? ` ·b${p.batch}` : "")] = (p[field] as number | null) ?? null;
    return row;
  });
}

const tipStyle = { background: "var(--bg-2)", border: "1px solid var(--line)", borderRadius: 10, color: "var(--fg)", fontSize: 13 };

function OneChart({ points, field, title, note, target, log }: { points: Point[]; field: keyof Point; title: string; note: string; target?: number | null; log?: boolean }) {
  const series = Array.from(new Set(points.map((p) => p.series + (p.batch > 1 ? ` ·b${p.batch}` : "")))).sort();
  const styles = seriesStyles(series);
  const rows = toRows(points, field);
  return (
    <div>
      <div className="small" style={{ fontWeight: 650 }}>{title} <span className="muted" style={{ fontWeight: 450 }}>— {note}</span></div>
      <ResponsiveContainer width="100%" height={290}>
        <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 28 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          <XAxis dataKey="conc" tick={{ fill: "var(--fg-3)", fontSize: 12 }} label={{ value: "requests at a time", position: "bottom", offset: 8, fill: "var(--fg-3)", fontSize: 12 }} />
          <YAxis tick={{ fill: "var(--fg-3)", fontSize: 12 }} scale={log ? "log" : "auto"} domain={log ? ["auto", "auto"] : [0, "auto"]} width={64} tickFormatter={(v) => fmt.n(v, log ? 2 : 0)} />
          <Tooltip contentStyle={tipStyle} formatter={(v: any) => fmt.n(Number(v), field === "cost" ? 2 : 0)} labelFormatter={(l) => `${l} concurrent`} />
          {target ? <ReferenceLine y={target} stroke="var(--bad)" strokeDasharray="6 4" label={{ value: "your target", fill: "var(--bad)", fontSize: 11, position: "insideTopRight" }} /> : null}
          {series.map((s) => (
            <Line key={s} type="monotone" dataKey={s} stroke={styles[s].color} strokeWidth={2.5} strokeDasharray={styles[s].dash}
              dot={(p: any) => (p.cx == null || p.cy == null || p.value == null ? <g key={`${s}-${p.index}`} /> : <Marker key={`${s}-${p.index}`} cx={p.cx} cy={p.cy} r={4} style={styles[s]} />)}
              activeDot={{ r: 6, fill: styles[s].color, stroke: "var(--bg-2)", strokeWidth: 2 }} connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function seriesKeys(points: Point[]): string[] {
  return Array.from(new Set(points.map((p) => p.series + (p.batch > 1 ? ` ·b${p.batch}` : "")))).sort();
}

/** One legend for the three charts, with the colour, dash and marker each line uses. */
export function ChartLegend({ points }: { points: Point[] }) {
  const series = seriesKeys(points);
  const styles = seriesStyles(series);
  return (
    <div className="chart-legend">
      {series.map((s) => (
        <span key={s} className="chart-legend-item"><Swatch style={styles[s]} />{s.startsWith("GPU") ? <b>GPU</b> : <b>CPU</b>} {s.replace(/^(GPU|CPU) · /, "")}</span>
      ))}
    </div>
  );
}

export function Curves({ points, units, target }: { points: Point[]; units: Group["units"]; target?: number | null }) {
  return (
    <>
      <ChartLegend points={points} />
      <div className="grid c3 mt">
        <OneChart points={points} field="throughput" title="Throughput" note={`${units.thr_label}, higher is better`} />
        <OneChart points={points} field="latency" title="Responsiveness" note={`${units.lat_label}, lower is better${target ? "; dashed line = your target" : ""}`} target={target} />
        <OneChart points={points} field="cost" title="Cost per unit" note={`${units.cost_label}, lower is better (log scale)`} log />
      </div>
    </>
  );
}

export function InstancesBar({ rows, limit }: { rows: Group["equivalence"]; limit: number }) {
  // "m8i.4xlarge · PyTorch · bf16 · 4 at a time": two CPU options on the same size differ by software only.
  const data = rows.map((r) => ({ name: `${r.series.replace(/^(GPU|CPU) · /, "")} · ${r.conc} at a time`, n: r.instances_per_gpu, ratio: r.hourly_cost_ratio, ok: r.within_limit }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(140, 36 * data.length + 70)}>
      <BarChart data={data} layout="vertical" margin={{ top: 22, right: 30, left: 10, bottom: 6 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={{ fill: "var(--fg-3)", fontSize: 12 }} allowDecimals={false} label={{ value: "CPU instances needed to match one GPU", position: "insideBottom", offset: -2, fill: "var(--fg-3)", fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={290} tick={{ fill: "var(--fg-2)", fontSize: 12 }} />
        <Tooltip contentStyle={tipStyle} formatter={(v: any, n: any, p: any) => [`${v} instances · ${fmt.x(p.payload.ratio)} the GPU's hourly cost`, n === "n" ? "Needed" : n]} />
        <ReferenceLine x={limit} stroke="var(--bad)" strokeDasharray="6 4" label={{ value: `limit ${limit}`, fill: "var(--bad)", fontSize: 11, position: "top" }} />
        <Bar dataKey="n" radius={[0, 6, 6, 0]} isAnimationActive={false}>
          {data.map((d, i) => <Cell key={i} fill={d.ok ? "var(--cpu)" : "var(--line-2)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}


// ----------------------------------------------------------------------------- utilization

/** How busy each hardware option's own processor (GPU or CPU) was at each load setting: the mean of
 * the 10-second samples over the measurement, with the 95th percentile as a thin line of the same colour. */
export function UtilizationCurves({ points }: { points: Point[] }) {
  const series = seriesKeys(points);
  const styles = seriesStyles(series);
  const xs = Array.from(new Set(points.map((p) => p.conc))).sort((a, b) => a - b);
  const rows = xs.map((x) => {
    const row: Record<string, number | string | null> = { conc: x };
    for (const p of points) if (p.conc === x) {
      const k = p.series + (p.batch > 1 ? ` ·b${p.batch}` : "");
      row[k] = p.util?.busy_mean ?? null;
      row[k + " (p95)"] = p.util?.busy_p95 ?? null;
    }
    return row;
  });
  const any = points.some((p) => p.util?.busy_mean != null);
  if (!any) return <div className="small muted mt">No utilization samples were recorded for these measurements.</div>;
  return (
    <div>
      <div className="small" style={{ fontWeight: 650 }}>How busy the hardware was <span className="muted" style={{ fontWeight: 450 }}>— GPU busy % for GPU options; for CPU options the share of the physical cores that were busy (the serving software runs one thread per core). Solid = average over the measurement, thin = 95th percentile. Shaded: below {35}% mostly idle, above {80}% at its limit.</span></div>
      <ResponsiveContainer width="100%" height={290}>
        <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 28 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          <ReferenceArea y1={0} y2={35} fill="var(--warn)" fillOpacity={0.07} />
          <ReferenceArea y1={80} y2={100} fill="var(--bad)" fillOpacity={0.07} />
          <XAxis dataKey="conc" tick={{ fill: "var(--fg-3)", fontSize: 12 }} label={{ value: "requests at a time", position: "bottom", offset: 8, fill: "var(--fg-3)", fontSize: 12 }} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--fg-3)", fontSize: 12 }} width={48} tickFormatter={(v) => `${v}%`} />
          <Tooltip contentStyle={tipStyle} formatter={(v: any) => `${fmt.n(Number(v), 0)}%`} labelFormatter={(l) => `${l} concurrent`} />
          {series.map((k) => (
            <Line key={k} type="monotone" dataKey={k} stroke={styles[k].color} strokeWidth={2.5} strokeDasharray={styles[k].dash}
              dot={(p: any) => (p.cx == null || p.cy == null || p.value == null ? <g key={`${k}-${p.index}`} /> : <Marker key={`${k}-${p.index}`} cx={p.cx} cy={p.cy} r={4} style={styles[k]} />)}
              activeDot={{ r: 6, fill: styles[k].color, stroke: "var(--bg-2)", strokeWidth: 2 }} connectNulls isAnimationActive={false} />
          ))}
          {series.map((k) => (
            <Line key={k + "p95"} type="monotone" dataKey={k + " (p95)"} stroke={styles[k].color} strokeWidth={1} strokeOpacity={0.55} dot={false} connectNulls isAnimationActive={false} legendType="none" />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const UTIL_LINES: { key: "cpu" | "gpu" | "mem" | "gpu_mem_pct"; label: string; color: string; dash?: string }[] = [
  { key: "gpu", label: "GPU busy %", color: "var(--gpu)" },
  { key: "cpu", label: "CPU busy % (whole host)", color: "var(--cpu)" },
  { key: "mem", label: "host memory used %", color: "var(--fg-3)", dash: "4 3" },
  { key: "gpu_mem_pct", label: "GPU memory used %", color: "var(--gpu)", dash: "2 4" },
];

/** One hardware option's utilization over the run (CloudWatch), with each measurement's window shaded
 * and labelled by its load setting, so idle gaps (model loading, waiting) read as such. */
export function UtilizationTimeline({ t }: { t: UtilTarget }) {
  const s = t.series;
  if (!s || !s.t.length) return <div className="small muted">{t.error ?? "CloudWatch has no utilization points for this hardware option in the run's window."}</div>;
  const t0 = s.t[0];
  const total = t.gpu_mem_total_mib;
  // CPU engines run one thread per physical core (two vCPUs each), so the host average tops out at 50%:
  // shown as the share of the physical cores, like the per-measurement figures (bench/utilization.cpu_scale).
  const cpuScale = t.accelerator === "cpu" && (t.spec?.vcpus ?? 0) > 1 ? 2 : 1;
  const cores = cpuScale > 1 ? Math.floor((t.spec?.vcpus ?? 0) / 2) : null;
  const rows = s.t.map((ts, i) => ({
    t: Math.round((ts - t0) / 60), cpu: s.cpu?.[i] != null ? Math.min(100, (s.cpu[i] as number) * cpuScale) : null, gpu: s.gpu?.[i] ?? null, mem: s.mem?.[i] ?? null,
    gpu_mem_pct: total && s.gpu_mem?.[i] != null ? Math.round((100 * (s.gpu_mem[i] as number)) / total) : null,
  }));
  const present = UTIL_LINES.filter((l) => rows.some((r) => r[l.key] != null)).map((l) => (l.key === "cpu" && cores ? { ...l, label: `CPU busy % (share of its ${cores} physical cores)` } : l));
  const windows = t.windows.map((w) => ({ ...w, x1: (w.start - t0) / 60, x2: (w.end - t0) / 60 }));
  return (
    <div>
      <div className="chart-legend">
        {present.map((l) => <span key={l.key} className="chart-legend-item"><svg width="26" height="10"><line x1="0" y1="5" x2="26" y2="5" stroke={l.color} strokeWidth="2.5" strokeDasharray={l.dash} /></svg>{l.label}</span>)}
        <span className="chart-legend-item"><span style={{ display: "inline-block", width: 14, height: 10, background: "var(--accent)", opacity: 0.18, borderRadius: 2 }} /> a measurement (hover for its load setting)</span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 28 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          {windows.map((w, i) => (
            <ReferenceArea key={i} x1={Math.round(w.x1)} x2={Math.max(Math.round(w.x1), Math.round(w.x2))} fill="var(--accent)" fillOpacity={0.12} ifOverflow="extendDomain"
              label={{ value: `${w.conc}${w.batch > 1 ? `×${w.batch}` : ""}`, position: "insideTop", fill: "var(--fg-3)", fontSize: 10 }} />
          ))}
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tick={{ fill: "var(--fg-3)", fontSize: 12 }} label={{ value: "minutes since the first measurement", position: "bottom", offset: 8, fill: "var(--fg-3)", fontSize: 12 }} />
          <YAxis domain={[0, 100]} tick={{ fill: "var(--fg-3)", fontSize: 12 }} width={48} tickFormatter={(v) => `${v}%`} />
          <Tooltip contentStyle={tipStyle} formatter={(v: any, name: any) => [`${fmt.n(Number(v), 0)}%`, UTIL_LINES.find((l) => l.key === name)?.label ?? name]} labelFormatter={(l) => `minute ${l}`} />
          {present.map((l) => <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} strokeWidth={l.dash ? 1.5 : 2.5} strokeDasharray={l.dash} dot={false} connectNulls isAnimationActive={false} />)}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
