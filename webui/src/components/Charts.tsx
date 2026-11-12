import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, type Group, type Point } from "../api";

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
