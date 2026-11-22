// User guide: docs/USER_GUIDE.md rendered as a document with a table of contents. The guide is part of the
// build (served by /api/guide), so what the reader sees describes the code that is running. Headings get
// stable ids from their text so sections can be linked (/guide#8-results) and the contents list can scroll
// to them; external links open in a new tab.
import { useEffect, useMemo, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api";
import { Note, Spinner } from "../components/ui";

/** A heading's anchor: lowercase words and digits joined by dashes ("8. Results" -> "8-results"). */
export function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

/** Plain text of a heading's children (react-markdown hands us strings and inline elements). */
function textOf(node: React.ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (typeof node === "object" && "props" in node) return textOf((node as any).props.children);
  return "";
}

/** The contents: every "## " and "### " line of the Markdown, in order (code fences excluded). */
export function contents(md: string): { level: 2 | 3; text: string; id: string }[] {
  const out: { level: 2 | 3; text: string; id: string }[] = [];
  let fenced = false;
  for (const line of md.split("\n")) {
    if (line.startsWith("```")) { fenced = !fenced; continue; }
    if (fenced) continue;
    const m = /^(##|###) (.+)$/.exec(line);
    if (m) out.push({ level: m[1].length as 2 | 3, text: m[2].trim(), id: slug(m[2]) });
  }
  return out;
}

export default function Guide() {
  const [md, setMd] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => { api.guide().then((g) => setMd(g.markdown)).catch((e) => setErr((e as Error).message)); }, []);
  const toc = useMemo(() => (md ? contents(md) : []), [md]);

  // Scroll to the section named in the URL once the text is in, and keep the contents list in step with scrolling.
  useEffect(() => {
    if (!md) return;
    const hash = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    if (hash) document.getElementById(hash)?.scrollIntoView();
    const heads = Array.from(document.querySelectorAll<HTMLElement>(".guide-body h2, .guide-body h3"));
    const obs = new IntersectionObserver((entries) => {
      const hit = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (hit) setActive(hit.target.id);
    }, { rootMargin: "0px 0px -70% 0px" });
    heads.forEach((h) => obs.observe(h));
    return () => obs.disconnect();
  }, [md]);

  const heading = (Tag: "h1" | "h2" | "h3" | "h4") => ({ children }: { children?: React.ReactNode }) => {
    const id = slug(textOf(children));
    return <Tag id={id}><a href={`#${id}`} className="anchor" aria-label="Link to this section">#</a>{children}</Tag>;
  };
  const components = useMemo(() => ({
    h1: heading("h1"), h2: heading("h2"), h3: heading("h3"), h4: heading("h4"),
    a: ({ href, children }: { href?: string; children?: React.ReactNode }) => {
      const external = !!href && /^https?:/.test(href);
      return <a href={href} target={external ? "_blank" : undefined} rel={external ? "noreferrer" : undefined}>{children}</a>;
    },
  }), []);

  if (err) return <div className="page"><Note kind="bad">The user guide could not be loaded: {err}</Note></div>;
  if (!md) return <div className="page"><Spinner label="Loading the user guide…" /></div>;
  return (
    <div className="page guide">
      <nav className="guide-toc card soft" aria-label="Contents">
        <div className="label">Contents</div>
        <ul>
          {toc.map((h) => (
            <li key={h.id} className={`lvl${h.level}${active === h.id ? " active" : ""}`}>
              <a href={`#${h.id}`} onClick={(e) => { e.preventDefault(); document.getElementById(h.id)?.scrollIntoView({ behavior: "smooth", block: "start" }); history.replaceState(null, "", `#${h.id}`); }}>{h.text}</a>
            </li>
          ))}
        </ul>
      </nav>
      <article className="card guide-body prose doc">
        <Markdown remarkPlugins={[remarkGfm]} components={components as any}>{md}</Markdown>
      </article>
    </div>
  );
}
