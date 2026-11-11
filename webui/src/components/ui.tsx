import type { ReactNode } from "react";

export function Help({ text }: { text: string }) {
  return (
    <span className="help" aria-label={text} role="img">
      ?
      <span className="tip">{text}</span>
    </span>
  );
}

export function Label({ children, help }: { children: ReactNode; help?: string }) {
  return (
    <div className="label">
      {children}
      {help && <Help text={help} />}
    </div>
  );
}

export function Badge({ kind, children }: { kind?: "gpu" | "cpu" | "ok" | "warn" | "bad" | "accent"; children: ReactNode }) {
  return <span className={"badge " + (kind ?? "")}>{children}</span>;
}

export function Note({ kind, children }: { kind?: "ok" | "warn" | "bad"; children: ReactNode }) {
  return <div className={"note " + (kind ?? "")}>{children}</div>;
}

export function Tile({ selected, onClick, title, desc, kind, badge, disabled, children }: {
  selected: boolean; onClick: () => void; title: ReactNode; desc?: ReactNode; kind?: "gpu" | "cpu"; badge?: ReactNode; disabled?: boolean; children?: ReactNode;
}) {
  return (
    <button type="button" className={`tile ${kind ?? ""} ${selected ? "selected" : ""}`} onClick={onClick} disabled={disabled}>
      <span className="check">{selected ? "✓" : ""}</span>
      <div className="t">{title} {badge}</div>
      {desc && <div className="d">{desc}</div>}
      {children}
    </button>
  );
}

export function Chips<T extends string | number>({ options, value, onChange, labels, disabled }: {
  options: T[]; value: T[]; onChange: (v: T[]) => void; labels?: (o: T) => string; disabled?: boolean;
}) {
  const toggle = (o: T) => onChange(value.includes(o) ? value.filter((v) => v !== o) : [...value, o].sort((a, b) => Number(a) - Number(b)));
  return (
    <div className="chips">
      {options.map((o) => (
        <button key={String(o)} type="button" className={"chip " + (value.includes(o) ? "on" : "")} onClick={() => toggle(o)} disabled={disabled}>
          {labels ? labels(o) : String(o)}
        </button>
      ))}
    </div>
  );
}

export function Toggle<T extends string>({ options, value, onChange }: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="toggle">
      {options.map((o) => (
        <button key={o.value} type="button" className={o.value === value ? "on" : ""} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Metric({ k, v, d, dGood, help }: { k: string; v: ReactNode; d?: ReactNode; dGood?: boolean | null; help?: string }) {
  return (
    <div className="metric">
      <div className="k">{k}{help && <Help text={help} />}</div>
      <div className="v">{v}</div>
      {d && <div className={"d " + (dGood === true ? "good" : dGood === false ? "badd" : "")}>{d}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return <span className="row small muted"><span className="spin" /> {label ?? "Loading…"}</span>;
}

export function Reasons({ reasons }: { reasons: { ok: boolean; text: string }[] }) {
  return (
    <div>
      {reasons.map((r, i) => (
        <div key={i} className={"reason " + (r.ok ? "ok" : "bad")}>
          <span className="ic">{r.ok ? "✓" : "!"}</span>
          <span>{r.text}</span>
        </div>
      ))}
    </div>
  );
}

export function instanceLine(s: { vcpus: number; memory_gib: number; gpus: number; gpu_model: string | null; gpu_memory_gib: number | null; usd_per_hour: number }, short = false) {
  const gpu = s.gpus ? (short ? ` · ${s.gpus}× ${s.gpu_model} ${Math.round(s.gpu_memory_gib ?? 0)} GiB` : ` · ${s.gpus}× ${s.gpu_model} (${Math.round(s.gpu_memory_gib ?? 0)} GiB GPU memory)`) : "";
  return `${s.vcpus} vCPU · ${s.memory_gib} GiB${gpu} · $${s.usd_per_hour.toFixed(2)}/h`;
}
