import { useState } from "react";
import { useCatalog } from "../App";
import DataPreview from "../components/DataPreview";
import { Tile } from "../components/ui";
import type { UseCase } from "../api";

export default function DataPage() {
  const cat = useCatalog();
  const [uc, setUc] = useState<UseCase>("standard");
  return (
    <div>
      <div className="topbar"><div><h1>Test data</h1><div className="muted">What the models are asked during a run. Each use case has its own data; nothing is downloaded or generated at run time.</div></div></div>
      <div className="tiles mb">
        {cat && (Object.entries(cat.use_cases) as [UseCase, any][]).map(([k, u]) => (
          <Tile key={k} selected={uc === k} onClick={() => setUc(k)} title={u.name} desc={u.short} />
        ))}
      </div>
      <div className="card"><DataPreview useCase={uc} /></div>
    </div>
  );
}
