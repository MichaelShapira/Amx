// Wraps the app in the cloud deployment: nothing renders until the user is signed in through the
// Cognito hosted login. Locally (/api/auth/config says enabled: false) it renders the app at once.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import type { User } from "oidc-client-ts";
import { authEnabled, completeSignIn, currentUser, isOperator, loadConfig, signIn, signOut, userLabel } from "../auth";
import { Note, Spinner } from "./ui";

export interface Session { enabled: boolean; user: User | null; label: string; operator: boolean; signOut: () => Promise<void> }
export const SessionCtx = createContext<Session>({ enabled: false, user: null, label: "", operator: true, signOut: async () => {} });
export const useSession = () => useContext(SessionCtx);

export default function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState<string>("");
  const loc = useLocation();
  const nav = useNavigate();

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const cfg = await loadConfig();
        if (!cfg.enabled) {
          if (alive) { setSession({ enabled: false, user: null, label: "", operator: true, signOut: async () => {} }); setState("ready"); }
          return;
        }
        if (!cfg.ready || !cfg.client_id) {
          // First minutes after a fresh deployment: the app client is registered after the service starts.
          if (alive) { setMessage("Sign-in is being set up for this deployment. Try again in a minute."); setState("error"); }
          return;
        }
        if (loc.pathname === "/callback") {
          const to = await completeSignIn();
          nav(to, { replace: true });
        }
        const user = await currentUser();
        if (!user) { await signIn(loc.pathname === "/callback" ? "/" : undefined); return; }
        if (alive) { setSession({ enabled: true, user, label: userLabel(user), operator: isOperator(user), signOut }); setState("ready"); }
      } catch (e) {
        if (alive) { setMessage(`Sign-in failed: ${(e as Error).message}`); setState("error"); }
      }
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (state === "loading") return <div className="main" style={{ padding: 40 }}><Spinner label={authEnabled() ? "Signing in…" : "Loading…"} /></div>;
  if (state === "error" || !session) {
    return (
      <div className="main" style={{ padding: 40, maxWidth: 640 }}>
        <h1>GPU vs AMX CPU benchmark</h1>
        <Note kind="bad">{message || "Sign-in failed."}</Note>
        <button className="btn" style={{ marginTop: 12 }} onClick={() => window.location.assign("/")}>Try again</button>
      </div>
    );
  }
  return <SessionCtx.Provider value={session}>{children}</SessionCtx.Provider>;
}
