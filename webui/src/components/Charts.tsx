import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, type Group, type Point } from "../api";

const GPU = ["#1f9d55", "#5cc98a", "#0e6b38", "#9ad9b5"];
const CPU = ["#2470d8", "#5f9bee", "#1446a0", "#2bb3c0", "#7f5af0", "#b48cff"];

export function seriesColors(series: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  let g = 0, c = 0;
  for (const s of series) out[s] = s.startsWith("GPU") ? GPU[g++ % GPU.length] : CPU[c++ % CPU.length];
  return out;
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
  const colors = seriesColors(series);
  const rows = toRows(points, field);
  return (
    <div>
      <div className="small" style={{ fontWeight: 650 }}>{title} <span className="muted" style={{ fontWeight: 450 }}>— {note}</span></div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
          <XAxis dataKey="conc" tick={{ fill: "var(--fg-3)", fontSize: 12 }} label={{ value: "concurrent requests", position: "insideBottom", offset: -2, fill: "var(--fg-3)", fontSize: 11 }} />
          <YAxis tick={{ fill: "var(--fg-3)", fontSize: 12 }} scale={log ? "log" : "auto"} domain={log ? ["auto", "auto"] : [0, "auto"]} width={64} tickFormatter={(v) => fmt.n(v, log ? 2 : 0)} />
          <Tooltip contentStyle={tipStyle} formatter={(v: any) => fmt.n(Number(v), field === "cost" ? 2 : 0)} labelFormatter={(l) => `${l} concurrent`} />
          {target ? <ReferenceLine y={target} stroke="var(--bad)" strokeDasharray="6 4" label={{ value: "your target", fill: "var(--bad)", fontSize: 11, position: "insideTopRight" }} /> : null}
          {series.map((s) => <Line key={s} type="monotone" dataKey={s} stroke={colors[s]} strokeWidth={2.5} dot={{ r: 4, strokeWidth: 2, fill: "var(--bg-2)" }} connectNulls isAnimationActive={false} />)}
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Curves({ points, units, target }: { points: Point[]; units: Group["units"]; target?: number | null }) {
  return (
    <div className="grid c3 mt">
      <OneChart points={points} field="throughput" title="Throughput" note={`${units.thr_label}, higher is better`} />
      <OneChart points={points} field="latency" title="Responsiveness" note={`${units.lat_label}, lower is better${target ? "; dashed line = your target" : ""}`} target={target} />
      <OneChart points={points} field="cost" title="Cost per unit" note={`${units.cost_label}, lower is better (log scale)`} log />
    </div>
  );
}

export function InstancesBar({ rows, limit }: { rows: Group["equivalence"]; limit: number }) {
  const data = rows.map((r) => ({ name: `${r.instance_type} · ${r.precision} · ${r.conc} conc`, n: r.instances_per_gpu, ratio: r.hourly_cost_ratio, ok: r.within_limit }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(120, 36 * data.length + 50)}>
      <BarChart data={data} layout="vertical" margin={{ top: 6, right: 30, left: 10, bottom: 6 }}>
        <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={{ fill: "var(--fg-3)", fontSize: 12 }} allowDecimals={false} label={{ value: "CPU instances needed to match one GPU", position: "insideBottom", offset: -2, fill: "var(--fg-3)", fontSize: 11 }} />
        <YAxis type="category" dataKey="name" width={240} tick={{ fill: "var(--fg-2)", fontSize: 12 }} />
        <Tooltip contentStyle={tipStyle} formatter={(v: any, n: any, p: any) => [`${v} instances · ${fmt.x(p.payload.ratio)} the GPU's hourly cost`, n === "n" ? "Needed" : n]} />
        <ReferenceLine x={limit} stroke="var(--bad)" strokeDasharray="6 4" label={{ value: `limit ${limit}`, fill: "var(--bad)", fontSize: 11, position: "top" }} />
        <Bar dataKey="n" radius={[0, 6, 6, 0]} isAnimationActive={false}>
          {data.map((d, i) => <Cell key={i} fill={d.ok ? "var(--cpu)" : "var(--line-2)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
