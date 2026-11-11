import { createContext, useContext, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api, type Catalog } from "./api";
import Home from "./pages/Home";
import NewRun from "./pages/NewRun";
import Runs from "./pages/Runs";
import Results from "./pages/Results";
import DataPage from "./pages/Data";

export const CatalogCtx = createContext<Catalog | null>(null);
export const useCatalog = () => useContext(CatalogCtx);

type Theme = "light" | "dark";

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem("theme") as Theme) ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [aws, setAws] = useState<{ ok: boolean; account?: string; identity?: string; error?: string } | null>(null);

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("theme", theme); }, [theme]);
  useEffect(() => { api.catalog().then(setCatalog).catch(() => setCatalog(null)); api.aws().then(setAws).catch(() => setAws({ ok: false, error: "API not reachable" })); }, []);

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
          </nav>
          <div className="spacer" />
          <div className={"status-pill " + (aws?.ok ? "" : "off")}>
            <span className="dot" /> {aws == null ? "Checking AWS…" : aws.ok ? `AWS ${aws.account} · ${aws.identity}` : "AWS not reachable"}
          </div>
          <button className="btn sm" style={{ marginTop: 10 }} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "☀️ Light mode" : "🌙 Dark mode"}
          </button>
          <div className="tiny" style={{ marginTop: 10 }}>Runs only on this machine (127.0.0.1). AWS calls use your own credentials.</div>
        </aside>
        <main className="main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/new" element={<NewRun />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:id" element={<Runs />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </main>
      </div>
    </CatalogCtx.Provider>
  );
}
