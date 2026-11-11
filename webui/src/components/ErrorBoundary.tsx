import { Component, type ReactNode } from "react";

/** A render error shows what went wrong instead of a blank page. */
export default class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div className="card" style={{ margin: 24 }}>
          <h2>Something went wrong on this page</h2>
          <p className="muted">The rest of the app still works. Reloading the page usually clears it; if it keeps happening, please report the message below.</p>
          <pre>{String(this.state.error?.message || this.state.error)}</pre>
          <button className="btn" onClick={() => this.setState({ error: null })}>Try again</button>
        </div>
      );
    }
    return this.props.children;
  }
}
