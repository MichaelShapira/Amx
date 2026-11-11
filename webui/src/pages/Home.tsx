import { Link } from "react-router-dom";
import { Badge } from "../components/ui";

export default function Home() {
  return (
    <div>
      <div className="hero">
        <h1>Should this model run on a GPU or on an Intel AMX CPU?</h1>
        <p>
          This tool answers that question with measurements, not opinions. It runs the same AI model on a GPU instance and on
          an AMX-capable CPU instance in your AWS account, under the same load, and tells you how fast each one is, what it
          costs, and how many CPU machines it takes to replace one GPU.
        </p>
        <div className="row" style={{ marginTop: 14 }}>
          <Link className="btn" to="/new">Start a benchmark →</Link>
          <Link className="btn outline" to="/results">See results</Link>
        </div>
      </div>

      <div className="grid c3">
        <div className="card">
          <h3>1 · Why</h3>
          <p className="muted">
            GPUs are scarce and come from a separate AWS quota pool. CPUs with Intel AMX (the matrix-math units in the
            m8i, c8i, m7i generations) can serve small models well. The question is whether they serve <em>your</em> model
            fast enough, and at what price.
          </p>
        </div>
        <div className="card">
          <h3>2 · How it works</h3>
          <p className="muted">
            You describe a test in a seven-step wizard. The tool starts the instances, loads the model on each, sends the
            same requests at the concurrency you chose, records every response time, and stops the instances. A report
            appears under <Link to="/results">Results</Link>.
          </p>
        </div>
        <div className="card">
          <h3>3 · What you get</h3>
          <p className="muted">
            A verdict per model in one sentence, head-to-head cards for the best GPU and best CPU measurement, charts across
            load levels, and for sensitive-data detection the accuracy on a dataset with known answers.
          </p>
        </div>
      </div>

      <div className="card mt">
        <h2>How to run a benchmark</h2>
        <ol className="muted" style={{ lineHeight: 1.8 }}>
          <li><b>Use case.</b> Chat and text generation, or sensitive-data detection (DSPM / DLP). Each has its own test data and its own success criteria.</li>
          <li><b>Test data.</b> See what the models will be asked. For chat: prompts of a chosen length. For detection: the scenarios (bulk scan, inline DLP, …), with real example documents and their expected answers.</li>
          <li><b>Quick check or full test.</b> A quick check takes under an hour and shows the shape of the result; it is labelled SMOKE and must not drive a decision. A full test collects 200 requests per level, three times, which is what the methodology needs for trustworthy percentiles.</li>
          <li><b>Models.</b> One or more; each runs on every hardware option so the comparison is like for like.</li>
          <li><b>Hardware.</b> Pick a GPU and a CPU option. Press <b>Propose</b> and the tool suggests instance sizes with the reason for each: does the model fit in memory, is the price in the fair band against the other side, how many cores feed AMX, is it a single GPU, has this size had capacity problems recently.</li>
          <li><b>Load.</b> Single request, N concurrent requests, or batched requests (for classifiers). Run the hardware options in parallel or one after another. Decide what counts as acceptable: how long until the first word, how long per generated word.</li>
          <li><b>Review and start.</b> The run is summarized in words with an estimated duration and cost before anything starts.</li>
        </ol>
      </div>

      <div className="grid c2 mt">
        <div className="card">
          <h2>How to read the findings</h2>
          <p className="muted">
            Everything is judged against <b>your targets</b> (for example "first word within 2 seconds for 99 of 100
            requests"). A measurement that meets them is a candidate; among candidates the report picks the fastest GPU and the
            fastest CPU.
          </p>
          <ul className="muted" style={{ lineHeight: 1.8 }}>
            <li><Badge kind="gpu">GPU</Badge> green and <Badge kind="cpu">CPU</Badge> blue, everywhere.</li>
            <li><b>Throughput</b>: how much work per second (tokens, documents or classifications). Higher is better; this is the capacity figure.</li>
            <li><b>Responsiveness</b>: the wait for 99 of 100 requests. Lower is better.</li>
            <li><b>Cost per unit</b>: the instance's hourly price divided by its throughput, as if held at that load for an hour. On-Demand prices always, so runs stay comparable.</li>
            <li><b>Instances per GPU</b>: how many of the CPU machine it takes to match one GPU's throughput, and the hourly cost of that many. Within your limit (default 4) means the CPU is a viable alternative; above it, conditional.</li>
            <li><b>AMX verified</b>: the tool checks that the CPU really used its AMX units (the matrix-math kernels in the engine trace). Rows that fail are marked INVALID and excluded.</li>
            <li><b>Quality loss</b> (chat) and <b>accuracy</b> (detection): speed is only half the story; the CPU's answers are compared with the GPU's, and detection is scored against the dataset's known answers with confidence intervals.</li>
          </ul>
        </div>
        <div className="card">
          <h2>Comparing runs</h2>
          <p className="muted">
            Under Results you can put several runs side by side, for example one model today and another next week. The tool
            re-checks every time whether the runs are comparable: same use case, same test data and answer length, same
            request shape (a single request, concurrent requests and batched requests load a machine differently), and GPU and
            CPU instances in a fair price band (within half to double of each other). Anything that is not comparable is
            blocked with the reason in words; small differences are shown as warnings.
          </p>
          <h3 className="mt">What runs cost you</h3>
          <p className="muted">
            Instances are launched when a run starts and stopped when it ends. Spot instances (the default) are cheaper spare
            capacity that AWS may reclaim; if that happens the run launches a replacement and re-runs only the unfinished
            measurements. A quick check on two instances costs about one dollar; a full test on all hardware options several
            hundred, shown before you start.
          </p>
        </div>
      </div>
    </div>
  );
}
