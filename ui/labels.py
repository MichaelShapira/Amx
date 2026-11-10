"""Plain-language names and help texts for the UI (Requirement 15). The reader is a product or
project manager deciding which instance type to buy: comfortable with "concurrent requests"
and "batch versus single run", not with engine flags or precision tags. Every control in the
app gets a help icon from here; the technical identifier is kept in parentheses so an
engineer can map it back to the config files."""

from __future__ import annotations

ACCOUNT_NOTE = ("Everything runs in your AWS account. Instances start when a run starts and are stopped or "
                "terminated when it ends; nothing runs between tests except the network gateway.")

SUITES = {
    "standard": ("Chat and text generation",
                 "Measures how fast each hardware option answers typical chat/assistant requests: time to the first "
                 "word, time per generated word, how many requests per second it sustains, and what that costs. "
                 "Also checks answer quality against the GPU."),
    "dspm": ("Sensitive-data detection (DSPM / DLP)",
             "Measures how fast and how accurately each hardware option classifies documents and messages as "
             "sensitive or not, using a synthetic dataset of 3,000 documents with known answers. Reports documents "
             "per second, cost per GB scanned, and detection accuracy with confidence intervals."),
}
SUITE_HELP = ("Pick the kind of workload you want to size hardware for. The two suites use different test data "
              "and different success criteria; a run covers one suite.")

MODELS = {
    "m1-qwen2.5-3b": ("Qwen2.5 3B Instruct", "General chat model, 3 billion parameters. Research license: results "
                                              "are for evaluation only, not for a commercial decision on this model."),
    "m2-qwen3-4b-2507": ("Qwen3 4B Instruct", "General chat model, 4 billion parameters, Apache-2.0 license."),
    "m3-roberta-large-mnli": ("RoBERTa-large (classifier)", "Text classifier, 0.36 billion parameters. Does not "
                                                            "generate text; answers 'does this text match this "
                                                            "statement' questions. Very fast, lower accuracy."),
    "m4-gemma-2-2b": ("Gemma 2 2B Instruct", "Google chat model, 2.6 billion parameters, Gemma license."),
    "m5-flan-t5-xl": ("Flan-T5 XL", "Instruction-following model of the older encoder-decoder type, 2.85 billion "
                                    "parameters. Included because many production classifiers still use this family."),
    "d1-privacy-filter": ("OpenAI privacy-filter (PII detector)", "Purpose-built detector, 1.4 billion parameters: "
                                                                  "marks the exact words that are personal data. "
                                                                  "Sensitive-data suite only."),
    "d2-roblox-pii-v2": ("Roblox PII classifier v2", "Purpose-built classifier, 0.56 billion parameters: flags "
                                                     "chat messages that contain personal data. Sensitive-data "
                                                     "suite only."),
}
MODELS_HELP = ("The AI models to test. Each selected model runs on every hardware option you pick below, so the "
               "comparison is like for like. Size (parameters) is the main driver of speed; the license says "
               "whether a commercial decision may rest on the result.")

ENGINES = {"vllm": "vLLM", "ovms": "OpenVINO", "thin-openvino": "OpenVINO", "thin-torch": "PyTorch"}
ENGINE_HELP = {
    "vllm": "vLLM: the most common open-source serving software for chat models, on GPUs and on Intel CPUs.",
    "ovms": "OpenVINO Model Server: Intel's optimized CPU serving software.",
    "thin-openvino": "OpenVINO through a small serving wrapper, for models the model server does not host.",
    "thin-torch": "Plain PyTorch through a small serving wrapper: the simplest, least optimized way to serve.",
}
HARDWARE_HELP = ("The hardware and serving software to compare. Tick every option you want in the comparison; "
                 "at least one GPU and one CPU option are needed for a verdict. Each option becomes one instance "
                 "that runs every selected model in turn. Pick the exact instance size per option; the price shown "
                 "is the public On-Demand rate used in the cost columns.")
INSTANCE_TYPE_HELP = ("The AWS instance size for this option. CPU options list only Intel generations with AMX, "
                      "the matrix-math feature this benchmark is about. The report prices results with the "
                      "On-Demand rate of the size that actually ran.")
FAIRNESS_HELP = ("A comparison is fair when the GPU and CPU instances cost about the same per hour (within half "
                 "to double). Comparing a $13/hour 8-GPU machine with a $0.85/hour CPU box would only show that "
                 "big machines are faster. The suggestions are the GPU sizes closest in price to your CPU choice.")

PRECISIONS = {
    "bf16": ("BF16 (standard)", "Full-quality number format used on GPUs and on AMX CPUs. The baseline."),
    "fp16": ("FP16 (half precision)", "Half-precision format; on CPUs it needs the AMX-FP16 feature. Same size as BF16, "
                                      "different rounding."),
    "w8a8": ("INT8 compressed (W8A8)", "Weights and activations stored as 8-bit integers: half the memory, faster on "
                                       "AMX, small quality loss. The report measures that loss."),
    "int8": ("INT8 compressed (OpenVINO)", "8-bit weights in Intel's OpenVINO format: smaller and faster, small quality "
                                           "loss."),
    "int4": ("INT4 compressed (OpenVINO)", "4-bit weights: smallest and fastest, larger quality loss."),
    "fp32": ("FP32 reference", "Full 32-bit precision. Slow; used only as the quality reference, not as a candidate."),
}
PRECISION_HELP = ("Number formats to test on this hardware. BF16 is the like-for-like baseline. Compressed formats "
                  "(INT8, INT4) are faster and cheaper on CPUs; the report shows what they cost in answer quality.")

CONCURRENCY_LEVELS = {1: "1 — one request at a time (single user)", 4: "4 — a small team",
                      8: "8", 16: "16 — a busy service", 32: "32", 64: "64 — a large service"}
CONCURRENCY_HELP = ("How many requests are kept in flight at the same time, per hardware option. Each level is a "
                    "separate measurement: the machine is loaded with exactly this many parallel requests until the "
                    "request budget is used up. Latency rises and throughput rises with concurrency; the verdict looks "
                    "for the level where the machine still meets your targets.")

DEPTH = {
    "quick": ("Quick check — 50 requests per level, 1 pass (not decision-grade)",
              "Enough to see the shape of the numbers in under an hour. Not enough for a decision: the report "
              "labels every number SMOKE."),
    "full": ("Full methodology — 200 requests per level, 3 passes",
             "The standard the methodology requires for a decision: percentiles from at least 200 requests, repeated "
             "three times so the report can show run-to-run variation."),
    "custom": ("Custom", "Set the request count and passes yourself."),
}
DEPTH_HELP = ("How much data each measurement collects. More requests give steadier percentiles; three passes show "
              "whether a result repeats.")
REQUESTS_HELP = "Requests sent at each concurrency level, per pass. The methodology needs at least 200 for a decision."
REPEATS_HELP = "How many times each level is repeated. Three passes let the report show run-to-run variation (CV)."

PROMPT_SETS = {
    "short": ("Short prompts (~128 tokens, a question)", "A sentence or two of input. Dominated by answer generation."),
    "medium": ("Medium prompts (~1,024 tokens, a page)", "About a page of input, as in summarizing an email thread."),
    "long": ("Long prompts (~4,096 tokens, a document)", "A long document as input. Dominated by reading (prefill) "
                                                        "speed, which differs a lot between CPU and GPU."),
    "custom": ("My own prompt", "Paste one prompt; every request uses it."),
}
PROMPT_SET_HELP = ("The length of the input text. A token is roughly three quarters of a word. Reading long input "
                   "and writing long output stress the hardware differently, so the report keeps them apart.")
MAX_NEW_HELP = ("How many tokens (roughly words) each answer generates. Every request generates exactly this many so "
                "that results are comparable; 128 is a short paragraph.")

SLA_HELP = {
    "ttft": ("First word within (ms, 99th percentile)", "The longest acceptable wait before the answer starts "
                                                        "appearing, for 99 out of 100 requests. 2,000 ms is a common "
                                                        "chat target. 0 = no limit."),
    "tpot": ("Time per generated word (ms, median)", "Typical time between two generated tokens; 150 ms is about "
                                                     "reading speed. 0 = no limit."),
    "e2e": ("Whole answer within (ms, 99th percentile)", "Optional overall limit per request. 0 = no limit."),
}
SLA_SECTION_HELP = ("The service levels a hardware option must meet to count as acceptable. The report marks every "
                    "measurement that meets them, then picks the cheapest CPU option and the best GPU option among "
                    "those. Capacity (can the CPU side match the GPU's throughput with a reasonable number of "
                    "machines) decides the verdict; cost is shown alongside.")

DSPM_WORKLOADS = {
    "W1": ("Bulk document scan: Sensitive / Not sensitive", "Classify whole document chunks (about 500 and 2,000 "
                                                            "tokens) as sensitive or not, as a data-store scanner does."),
    "W2": ("Bulk document scan: which category", "Same documents, but name the category: government ID, personal, "
                                                 "financial, health, credentials, confidential, or none."),
    "W3": ("Inline DLP: Allow / Block a message", "Chat messages, emails and AI prompts (short), decided in real time "
                                                  "before they leave the company."),
    "W4": ("Database columns: which category", "A table header plus sample rows; name the sensitive category of the "
                                               "table, or none."),
}
DSPM_WORKLOADS_HELP = ("The detection scenarios to run. Each uses its own slice of the synthetic dataset and its own "
                       "expected answers; speed is reported per scenario.")
DSPM_PRESETS = {
    "inline_strict": ("Inline, strict — every decision within 250 ms", "For blocking a message before it is sent, "
                                                                      "where the user is waiting."),
    "inline_standard": ("Inline, standard — every decision within 1 second", "For real-time checks where a one-second "
                                                                            "pause is acceptable."),
    "bulk": ("Bulk scan — within 30 seconds, maximize documents per second", "For scanning data at rest, where "
                                                                            "throughput matters and latency does not."),
}
DSPM_PRESET_HELP = ("What counts as fast enough for the sensitive-data suite. 'Within' means the 99th-percentile "
                    "decision time. Pick the preset that matches how the detector would be used.")
DSPM_ITEMS_HELP = ("How many of the 3,000 test documents each option classifies for the accuracy figures. 300 gives "
                   "a quick read with wide confidence intervals; the full 3,000 gives the decision-grade figures.")

PURCHASE = {
    "spot": ("Spot — cheaper spare capacity, may be interrupted", "AWS may reclaim a Spot instance with two minutes' "
                                                                  "notice. If that happens the run launches a "
                                                                  "replacement and re-runs only the unfinished "
                                                                  "measurements; the report notes the interruption. "
                                                                  "Cost columns always use On-Demand prices so results "
                                                                  "stay comparable."),
    "on-demand": ("On-Demand — guaranteed for the run, full price", "Regular instances. Use when Spot keeps getting "
                                                                    "interrupted or has no capacity."),
}
PURCHASE_HELP = "How the test instances are bought. This only affects what the run costs you, not the results."
FALLBACK_HELP = ("If no Spot capacity is available for 10 minutes, launch an On-Demand instance instead so the run "
                 "does not stall.")
RUN_NAME_HELP = "A short label for this run; it appears in the run list and in file names."

RESULT_COLUMNS = {
    "model": ("Model", "The AI model tested."),
    "target": ("Hardware option", "The hardware and serving software option (see the run definition)."),
    "instance_type": ("Instance", "The AWS instance size that actually ran."),
    "engine": ("Software", "Serving software: vLLM, OpenVINO or PyTorch."),
    "precision": ("Number format", "BF16 is the baseline; INT8/INT4 are compressed formats."),
    "workload": ("Scenario", "Sensitive-data scenario (W1 bulk scan, W3 inline DLP, ...), blank for chat runs."),
    "conc": ("Concurrent requests", "How many requests were in flight at once during this measurement."),
    "batch": ("Batch size", "Inputs per request for classifiers; 1 for chat models."),
    "e2e_p50": ("Whole answer, median (ms)", "Half of the requests finished within this time."),
    "e2e_p99": ("Whole answer, 99th pct (ms)", "99 of 100 requests finished within this time."),
    "ttft_p99": ("First word, 99th pct (ms)", "Wait before the answer started appearing, for 99 of 100 requests."),
    "tpot_p50": ("Per generated word, median (ms)", "Typical time between two generated tokens."),
    "agg_out_tok_s": ("Throughput (tokens/s)", "Generated tokens per second across all concurrent requests. Higher is "
                                               "better; this is the capacity figure."),
    "docs_s": ("Documents/s", "Documents classified per second (sensitive-data suite)."),
    "cls_s": ("Classifications/s", "Inputs classified per second (classifier models)."),
    "usd_per_1m_tokens": ("$ per 1M tokens", "Instance price divided by throughput, as if held at this load for an hour."),
    "usd_per_gb": ("$ per GB scanned", "Cost to classify one GB of text at this rate (sensitive-data suite)."),
    "amx": ("AMX verified", "VERIFIED: the CPU really used its AMX matrix units; INVALID rows are excluded from the "
                            "verdict; n/a for GPUs."),
    "quality_delta": ("Quality loss vs GPU", "1 minus the similarity of this option's answers to the GPU's answers "
                                             "(0 = identical). BF16 on CPU sets the noise floor."),
    "meets_user_sla": ("Meets targets", "Whether this measurement meets the service levels set above."),
    "user_sla_reason": ("Why not", "The first target it missed, if any."),
}
CHARTS = {"e2e_p99": "Whole answer, 99th percentile (ms)", "e2e_p50": "Whole answer, median (ms)",
          "ttft_p99": "First word, 99th percentile (ms)", "agg_out_tok_s": "Throughput (tokens/s)",
          "usd_per_1m_tokens": "$ per 1M tokens", "docs_s": "Documents/s", "usd_per_gb": "$ per GB scanned",
          "cls_s": "Classifications/s"}
CAPACITY_HELP = ("How many CPU instances it takes to deliver the same throughput as the best GPU measurement that "
                 "meets your targets, and what that many CPUs cost per hour relative to the GPU. 'Standard vCPUs' "
                 "and 'G vCPUs' are the two separate AWS quota pools the instances draw from: that is the capacity "
                 "argument in one number.")

CALC_HELP = {
    "name": "A label for this model in the estimate.",
    "type": ("decoder = chat/generation model; encoder-decoder = Flan-T5 style; encoder-classifier = text classifier; "
             "token-classifier = marks words in a text (PII detector)."),
    "params": "Model size in billions of parameters.",
    "stack": "How the model is served today on GPUs.",
    "cuda": "GPU-only libraries the current deployment depends on; each one is work to replace on CPU.",
    "precision": "The number format you would run on CPU. Compressed formats add conversion and validation work.",
    "conc": "The concurrency the service must sustain, if known; used with a benchmark summary for the recommendation.",
    "infra": "Where the service runs; affects the integration work.",
    "iac": "Whether infrastructure-as-code and a CI/CD pipeline already exist for the service.",
    "familiarity": "How much experience the team has tuning CPU inference.",
    "summary": "A summary.json from the Results page. With it, the calculator adds a hardware recommendation per model.",
}

DATA_INTRO = {
    "standard": ("The chat suite sends fixed prompts built from public-domain books (Project Gutenberg), cut to a "
                 "target length and wrapped in each model's own chat format, so every model reads the same text. "
                 "Answers are generated to a fixed length so that timings are comparable. A separate set of 50 "
                 "real instructions (databricks-dolly-15k) and 50 sentence pairs (SNLI) checks answer quality "
                 "against the GPU."),
    "dspm": ("The sensitive-data suite uses a synthetic dataset built for this benchmark: 3,000 test documents and "
             "300 development documents with known answers. About half contain sensitive data (government IDs, "
             "personal details, financial data, health data, credentials, confidential business content); the rest "
             "do not, and about a third of those are hard negatives that look sensitive but are not (masked card "
             "numbers, documentation placeholders, public business data, first names alone). Sources: NVIDIA "
             "Nemotron-PII (CC BY 4.0), Gretel synthetic PII finance (Apache-2.0) and a seeded generator in this "
             "repository. All values are synthetic; the dataset is kept out of git because scanners flag the fake "
             "secrets."),
}
DATA_HELP = {
    "set": "Which prompt set to look at; the run uses the one chosen under 'Input length'.",
    "model": "Prompts are wrapped in each model's chat format, so the exact text differs slightly per model.",
    "scenario": "Which detection scenario's documents to show.",
    "label": "Expected answer: what the detector should say about this document.",
    "hard_case": "Hard negatives: documents that look sensitive but are not; the report shows false positives per type.",
    "sample": "A random sample from the test set; press again for other examples.",
    "policy": "The classification rules the models are given. Its fingerprint is recorded with every run.",
}

GRADE_HELP = ("Decision-grade: at least 200 requests per concurrency level and 3 passes, the minimum the methodology "
              "requires before percentiles are trustworthy. Quick check: anything less; it exercises the whole pipeline "
              "and shows the shape of the result, but the report labels every number SMOKE (the engineering term for a "
              "pipeline check) and it should not drive a buying decision.")
