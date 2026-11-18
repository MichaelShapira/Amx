import { createContext, useContext, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api, type Catalog, type AwsStatus } from "./api";
import Home from "./pages/Home";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import Results from "./pages/Results";
import DataPage from "./pages/Data";
import Setup from "./pages/Setup";
import Technical from "./pages/Technical";
import ErrorBoundary from "./components/ErrorBoundary";
import AuthGate, { useSession } from "./components/AuthGate";

export const CatalogCtx = createContext<Catalog | null>(null);
export const useCatalog = () => useContext(CatalogCtx);

type Theme = "light" | "dark";

export default function App() {
  return <AuthGate><Shell /></AuthGate>;
}

function Shell() {
  const session = useSession();
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem("theme") as Theme) ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [aws, setAws] = useState<AwsStatus | null>(null);

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("theme", theme); }, [theme]);
  useEffect(() => { api.catalog().then(setCatalog).catch(() => setCatalog(null)); }, []);
  // The operator's AWS session lasts about an hour. Re-check every 30 s so the pill turns red with
  // the reason when it expires, and green again once fresh credentials are in place (no restart).
  useEffect(() => {
    const check = () => api.aws().then(setAws).catch(() => setAws({ ok: false, error: "API not reachable" }));
    check(); const t = setInterval(check, 30000); return () => clearInterval(t);
  }, []);

  return (
    <CatalogCtx.Provider value={catalog}>
      <div className="app">
        <aside className="sidebar">
          <div className="brand">
            <div className="logo">⚡</div>
            <div><div className="name">GPU vs AMX CPU</div><div className="sub">inference benchmark</div></div>
          </div>
          <nav className="nav">
            <NavLink to="/" end><span className="ico">🏠</span> Start here</NavLink>
            <NavLink to="/new"><span className="ico">🧪</span> New benchmark</NavLink>
            <NavLink to="/data"><span className="ico">📚</span> Test data</NavLink>
            <NavLink to="/runs"><span className="ico">⏱️</span> Runs</NavLink>
            <NavLink to="/results"><span className="ico">📊</span> Results</NavLink>
            <NavLink to="/setup"><span className="ico">⚙️</span> Setup</NavLink>
            <NavLink to="/technical"><span className="ico">🧩</span> Technical details</NavLink>
          </nav>
          <div className="spacer" />
          <div className={"status-pill " + (aws?.ok ? "" : "off")} title={aws?.ok ? `Credentials: ${aws.credentials ?? ""}` : aws?.error ?? ""}>
            <span className="dot" /> {aws == null ? "Checking AWS…" : aws.ok ? `AWS ${aws.account} · ${aws.identity}` : aws.expired ? "AWS session expired" : "AWS not reachable"}
          </div>
          {aws && !aws.ok && aws.error && <div className="tiny" style={{ marginTop: 6, color: "var(--bad)" }}>{aws.error}</div>}
          <button className="btn sm" style={{ marginTop: 10 }} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "☀️ Light mode" : "🌙 Dark mode"}
          </button>
          {session.enabled ? (
            <div className="tiny" style={{ marginTop: 10 }}>
              Signed in as <b>{session.label}</b>{session.operator ? "" : " (view only)"}.
              <button className="btn sm" style={{ marginTop: 8, display: "block" }} onClick={() => void session.signOut()}>Sign out</button>
            </div>
          ) : (
            <div className="tiny" style={{ marginTop: 10 }}>Runs only on this machine (127.0.0.1). AWS calls use your own credentials.</div>
          )}
        </aside>
        <main className="main">
          <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/new" element={<NewRun />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:id" element={<Runs />} />
            <Route path="/results" element={<Results />} />
            <Route path="/setup" element={<Setup />} />
            <Route path="/technical" element={<Technical />} />
            <Route path="/callback" element={<Home />} />
          </Routes>
          </ErrorBoundary>
        </main>
      </div>
    </CatalogCtx.Provider>
  );
}
