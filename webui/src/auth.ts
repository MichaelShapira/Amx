// Sign-in for the cloud deployment: Amazon Cognito hosted login, authorization code + PKCE, through
// oidc-client-ts. The API tells the browser the pool, client id and hosted-login domain
// (/api/auth/config, public). Tokens live in sessionStorage for this tab; the access token goes on
// every /api call as a Bearer token and is renewed with the refresh token before it expires.
// Locally (enabled: false) nothing here runs and the API is open on loopback.
import { User, UserManager, WebStorageStateStore } from "oidc-client-ts";

export interface AuthConfig {
  enabled: boolean; ready?: boolean; region?: string; user_pool_id?: string; issuer?: string;
  client_id?: string | null; domain?: string; operator_group?: string;
}

let config: AuthConfig | null = null;
let manager: UserManager | null = null;
let redirecting = false;

export async function loadConfig(): Promise<AuthConfig> {
  if (config) return config;
  const r = await fetch("/api/auth/config", { cache: "no-store" });
  if (!r.ok) throw new Error(`auth config: HTTP ${r.status}`);
  config = (await r.json()) as AuthConfig;
  return config;
}

function userManager(): UserManager {
  if (manager) return manager;
  if (!config?.enabled || !config.client_id || !config.issuer) throw new Error("sign-in is not configured");
  manager = new UserManager({
    authority: config.issuer,                       // OIDC discovery: <issuer>/.well-known/openid-configuration
    client_id: config.client_id,
    redirect_uri: `${window.location.origin}/callback`,
    post_logout_redirect_uri: `${window.location.origin}/`,
    response_type: "code",
    scope: "openid email profile",
    automaticSilentRenew: true,                      // uses the refresh token; no iframe
    userStore: new WebStorageStateStore({ store: window.sessionStorage }),
    revokeTokensOnSignout: false,
  });
  return manager;
}

export const authEnabled = () => Boolean(config?.enabled);

export async function currentUser(): Promise<User | null> {
  if (!authEnabled()) return null;
  const u = await userManager().getUser();
  return u && !u.expired ? u : null;
}

/** Start the hosted sign-in, remembering where to come back to. Idempotent per page load. */
export async function signIn(returnTo = window.location.pathname + window.location.search): Promise<void> {
  if (redirecting) return;
  redirecting = true;
  await userManager().signinRedirect({ state: { returnTo } });
}

/** Finish the sign-in on /callback; returns the path to continue at. */
export async function completeSignIn(): Promise<string> {
  const u = await userManager().signinRedirectCallback();
  const st = u.state as { returnTo?: string } | undefined;
  const to = st?.returnTo && st.returnTo.startsWith("/") && !st.returnTo.startsWith("//") ? st.returnTo : "/";
  return to === "/callback" ? "/" : to;
}

/** Cognito's sign-out endpoint takes client_id + logout_uri (not the OIDC end_session shape). */
export async function signOut(): Promise<void> {
  const um = userManager();
  await um.removeUser();
  const q = new URLSearchParams({ client_id: config!.client_id!, logout_uri: `${window.location.origin}/` });
  window.location.assign(`${config!.domain}/logout?${q}`);
}

export async function accessToken(): Promise<string | null> {
  const u = await currentUser();
  return u?.access_token ?? null;
}

export function userLabel(u: User | null): string {
  const p = u?.profile as Record<string, unknown> | undefined;
  return (p?.email as string) || (p?.["cognito:username"] as string) || (p?.username as string) || "signed in";
}

export function isOperator(u: User | null): boolean {
  const groups = (u?.profile as Record<string, unknown> | undefined)?.["cognito:groups"] as string[] | undefined;
  return !config?.enabled || (groups ?? []).includes(config.operator_group ?? "operators");
}
