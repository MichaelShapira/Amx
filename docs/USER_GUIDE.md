# User guide

This guide is for the people who have to make the decision: product managers, finance, and anyone who owns an AI feature's bill. It explains why the tool exists, what it measures, every screen and every button, and how to read a result without being an engineer. Engineers will want the **Technical details** page as well, which shows the method down to the exact container images and scripts.

If you read nothing else: a benchmark takes seven clicks to set up, runs unattended in your own AWS account, and ends in a page whose first section is called **Where the CPU is the better choice**. That section is the answer. Everything else on the page is the evidence behind it.

---

## 1. The problem this tool solves

### 1.1 The question

Every AI feature runs on hardware that someone pays for by the hour. For the small and mid-sized models that most product features use (chat assistants of a few billion parameters, classifiers, embedding models, decision models inside agents) there are two kinds of machine on AWS:

- **GPU instances** (NVIDIA cards such as the L4 in `g6` or the L40S in `g6e`). The default choice. Very fast, but scarce: GPU capacity sits in a separate AWS quota, launches are refused when a region is busy, and the smallest GPU machine still costs about a dollar an hour whether it is busy or idle.
- **CPU instances with Intel AMX** (the `m8i`, `c8i` and `m7i` generations of Intel Xeon). AMX, Advanced Matrix Extensions, is a matrix-math unit built into every core of these processors since Sapphire Rapids (2023). Intel publishes benchmarks showing these CPUs serving small models at useful speeds. They come in small sizes, from a few dollars a day, and are the same general-purpose fleet many teams already run.

The question a product team faces is therefore not "which is faster" (the GPU is, almost always) but:

> **For my model, my test data and my load, does the CPU deliver enough, and what does each option cost me per unit of work?**

### 1.2 Why opinions are not enough

Vendor numbers answer a different question than yours. Intel measures two-socket servers with 128 cores; a cloud slice has 8 or 16. NVIDIA measures its best case. Both measure their own prompts, their own answer lengths and their own definition of "fast enough". Your model, your prompt lengths, your concurrency and your latency target change the answer by an order of magnitude.

The honest way to decide is to run **your** model with **your** data on both kinds of machine under **the same load**, and to look at the result against **your** targets. This tool does that, in your AWS account, and writes down everything it did so the result can be checked.

### 1.3 What the tool concluded so far, so you know what to expect

Across every model measured at full grade in this deployment, one pattern holds:

- **On peak throughput per dollar the GPU wins.** On AWS pricing, one `g6.xlarge` (about $0.80/h) delivers roughly 4 to 15 times the work of a price-matched `m8i.4xlarge` (about $0.85/h) for the same model, depending on the task: 4.3x on embeddings (run `embedding-full-e1-20261119T141541Z`, 2026-11-19), 7.3x on RoBERTa classification (`roberta-int8-full-20261120T211803Z`, 2026-11-20), 14 to 16x on inline DLP and short-document scans with Gemma 4 E4B (`gemma-bulk-scan-dlp-dspm-20261122T085327Z`, 2026-11-22), about 4x on chat (`docs/recipe-check-2026-11-19.md`). The one measured exception so far: long-document scans with Gemma 4 E4B, where the CPU matched the GPU (same run). This is physics and pricing, not software: the L4 has about 7.5 times the arithmetic (121 dense BF16 TFLOPS, NVIDIA's L4 datasheet, against 8 cores at about 2 TFLOPS of AMX BF16 each, Intel's 1,024 FLOPs per cycle per core) and 2.7 times the memory bandwidth (300 GB/s against 107 GB/s measured with STREAM on the `m8i.4xlarge`) of the 8 cores you get for the same money, and the CPU side here runs at 70 to 80 per cent of its own hardware limit with AMX verified (`docs/recipe-check-2026-11-19.md`). Every figure in this paragraph is in the run's report under Results, or in that document; see also section 1.5.
- **The CPU wins in four situations**, which the results page computes for every run:
  1. **At low or moderate load**, because the smallest CPU slice that meets your targets is far cheaper than the smallest GPU you can buy. Below a load the page calls the **crossover**, the CPU is the cheaper option.
  2. **On availability**, when AWS refuses to launch the GPU size you want and launches the CPU on request. The page counts the refusals from the run's own log.
  3. **On the smallest unit you can buy**: a GPU is at least $0.80/h, idle or not; a CPU slice that fits a small model can be $0.10/h.
  4. **On capacity you already pay for**: an existing CPU fleet with idle cycles costs close to nothing extra. The page lets you enter your own hourly rate, including zero.

So the tool is not trying to prove that the CPU is faster. It measures where each option is the better buy, and says so in plain words.

### 1.4 Where to read more (external references)

- Intel AMX overview: <https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html>
- Intel's published Xeon inference figures (Software Catalog, the method the "Intel's published benchmark" use case reproduces): <https://www.intel.com/content/www/us/en/developer/tools/software-catalog/overview.html>
- AWS instance families used here: M8i <https://aws.amazon.com/ec2/instance-types/m8i/>, C8i <https://aws.amazon.com/ec2/instance-types/c8i/>, G6 (NVIDIA L4) <https://aws.amazon.com/ec2/instance-types/g6/>, G6e (NVIDIA L40S) <https://aws.amazon.com/ec2/instance-types/g6e/>
- AWS On-Demand pricing and Savings Plans: <https://aws.amazon.com/ec2/pricing/on-demand/> and <https://aws.amazon.com/savingsplans/compute-pricing/>
- vLLM, the serving software on both sides: <https://docs.vllm.ai/>, and its Intel CPU backend <https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html>
- Strands Decider, the decision model family: <https://huggingface.co/StrandsAgents> and <https://github.com/strands-labs/strands-decider>
- Hugging Face model hub, where every model and its exact version come from: <https://huggingface.co/>
- Amazon Bedrock (the reasoning model that proposes hardware and writes the executive summary): <https://aws.amazon.com/bedrock/>

### 1.5 Sources for the figures quoted in this guide

| Figure | Source |
|---|---|
| L4: 121 dense BF16 TFLOPS, 300 GB/s | NVIDIA L4 product page and datasheet, <https://www.nvidia.com/en-us/data-center/l4/>; the same numbers drive the "Why it came out this way" card (`bench/mechanics.py`). |
| AMX: 1,024 BF16 FLOPs per cycle per core | Intel AMX overview (link above); about 2 TFLOPS per Xeon 6 core at AMX clock, the figure `bench/mechanics.py` uses. |
| m8i.4xlarge memory bandwidth 107 GB/s | Measured with STREAM on every CPU test host by the benchmark itself (the "membw" probe); shown on each run's "Why it came out this way" card. |
| CPU at 70 to 80 per cent of its hardware limit | `docs/recipe-check-2026-11-19.md`, runs `recipe-check-20261119T081855Z` and `recipe-check-20261119T085351Z`: the harness's vLLM configuration against Intel's own recipe on the same instance, decode at about 70 per cent of measured bandwidth, per-core throughput inside Intel's published range. |
| GPU/CPU ratios 4.3x, 7.3x, 14 to 16x, about 4x | The runs named in section 1.3, each a full-grade report under Results in this deployment, and the recipe-check document for chat cost per million tokens. |
| Spot 30 to 70 per cent below On-Demand | AWS Spot Instance Advisor savings for the `g6`, `g6e` and `m8i` sizes in us-east-1 at the time of writing (November 2026), <https://aws.amazon.com/ec2/spot/instance-advisor/>; AWS states savings of up to 90 per cent, <https://aws.amazon.com/ec2/spot/>. The estimate shown before a run uses On-Demand prices on purpose. |
| Savings Plan rates | AWS Price List API (`savingsplans describe-savings-plans-offering-rates`), snapshotted into the instance catalog with its date; shown next to every price on the Results page. |
| Instance prices | AWS Price List API, On-Demand, Linux, shared tenancy, snapshotted with the catalog; the snapshot date is printed on the Results page. |

---

## 2. How the tool works, in one page

1. You describe a test in a **seven-step wizard**: use case, test data, depth, model, hardware, load and targets, review.
2. When you press **Start**, the tool launches **one EC2 instance per hardware option** in your AWS account (Spot by default, cheaper spare capacity), downloads the model at a pinned version, starts the serving software, and checks that it answers.
3. It then sends the **same requests** to each machine at each load setting you chose, records every response time and every answer, and repeats as many times as you asked.
4. On CPU machines it **proves that AMX was used** by reading the processor's own counters during the measurements and the serving software's kernel trace. A CPU result that did not run on AMX is marked INVALID.
5. It samples how busy each machine was every 10 seconds, so the report can say whether the machine was the right size.
6. For use cases with known answers (sensitive-data detection, translation, embeddings, decisions) it also scores **accuracy or quality**, and compares the CPU's answers with the GPU's.
7. It stops the instances, writes a report, and the **Results** page turns the report into verdicts, cards and charts.

A quick check of one model on two machines takes about 45 minutes and costs around a dollar. A full, decision-grade test takes several hours and costs tens of dollars; the exact estimate is shown before anything starts.

Nothing is installed on your laptop. The tool is a web application in your account, protected by a sign-in; everything it does is logged and can be read on the **Technical details** page.

---

## 3. Getting in

### 3.1 Signing in

Open the application URL. You are sent to the sign-in page (AWS Cognito). Enter the e-mail address and the password you were given. There is no self-registration: an administrator creates each account.

After sign-in you land on **Start here**. The left-hand sidebar shows:

| Menu item | What it is for |
|---|---|
| **Start here** | A one-page overview and the "Start a benchmark" button. |
| **New benchmark** | The seven-step wizard. |
| **Test data** | What the models are asked, per use case, with real examples. |
| **Runs** | Every benchmark, running or finished, with its live progress. |
| **Results** | Reports, verdicts and charts; compare several runs side by side. |
| **Setup** | The model catalog, the Hugging Face token, the reasoning model. |
| **Technical details** | The method for engineers: architecture, steps, containers, files. |
| **User guide** | This document. |

At the bottom of the sidebar: the AWS account the tool is running in, a **Dark mode** toggle, who is signed in, and **Sign out**.

### 3.2 Two kinds of user

- **Readers** can open every page, look at results and download reports.
- **Operators** can also start, stop and delete runs, change Setup and generate executive summaries (each summary is a paid call to Amazon Bedrock). If a button tells you that you lack permission, ask an administrator to add you to the operators group.

### 3.3 When the session expires

Sign-in lasts a limited time. When it runs out, the sidebar shows a pill saying so and the next click sends you back to the sign-in page; nothing you had on screen is lost on the server side. The wizard remembers its draft in your browser.

---

## 4. Start here

The landing page is a condensed version of this guide: why, how it works, what you get, how to run a benchmark, how to read the findings, how runs are compared, and what runs cost. Two buttons: **Start a benchmark →** opens the wizard; **See results** opens Results.

The colour convention used everywhere: **GPU is green, CPU is blue.**

---

## 5. New benchmark: the wizard, step by step

The wizard has seven numbered steps across the top. You can click any step to jump to it; nothing starts until you press **Start the benchmark** on the last one. Your draft is kept in the browser, so you can leave and come back.

If you arrived here from a finished run's **Copy to new run** or **Re-run with recommended sizes**, a green note at the top says which run the settings were copied from and what was changed.

### 5.1 Step 1: What do you want to size hardware for? (use case)

Pick one tile. The use case decides the test data, the units in the report and which targets apply.

| Use case | What it measures | Unit in the report |
|---|---|---|
| **Chat and text generation** | Time to the first word, time per generated word, requests per second, cost per million generated tokens, and answer quality against the GPU. For chatbots, summarisation, drafting. | tokens per second |
| **Sensitive-data detection (DSPM / DLP)** | How fast and how accurately each option flags sensitive documents and messages, on a dataset of 3,000 documents with known answers. For data-store scanners and data-loss-prevention gates. | documents per second, $ per GB scanned |
| **Translation (EN ⇄ DE, EN ⇄ ES)** | Characters translated per second, cost per million characters, time to the first word and to the whole translation, and translation quality (chrF) against a reference and against the GPU. | characters per second |
| **Embeddings (search, RAG, deduplication)** | Embeddings per second, cost per million texts, whether the CPU's vectors match the GPU's, and whether a German or Spanish text retrieves its English translation. | embeddings per second |
| **Agent decisions (routing, guardrails, triage)** | A decision model picks one of N options, answers yes/no or rates on a scale with a calibrated confidence. Decisions per second, cost per million decisions, accuracy against gold answers, agreement with the GPU. | decisions per second |
| **Intel's published benchmark (reproduced)** | The exact test Intel's Software Catalog publishes Xeon numbers for, with Intel's figures shown beside yours per core. | tokens per second |

Choosing a use case also fills in the **typical tests** for it (load settings and targets a first benchmark should use); you can change them in step 6.

### 5.2 Step 2: Which test data?

Shows what the models will be asked, with a **Preview** of real examples. The fields depend on the use case:

- **Chat**: the **prompt set** (Short prompts, about 128 tokens, a question or two sentences; Medium, about 1,024 tokens, a page; Long, about 4,096 tokens, a long document, dominated by reading speed which differs most between CPU and GPU) and the **Answer length (tokens)**. Every request generates exactly this many tokens so results are comparable; 128 is a short paragraph.
- **Sensitive-data detection**: the **scenarios** (called workloads): Bulk document scan, Sensitive / Not sensitive; Bulk document scan, which category; Inline DLP, Allow / Block a message; Database columns, which category. And **Documents in the accuracy check**: how many of the 3,000 test documents each option classifies. 300 gives a quick read with wide confidence intervals; 3,000 gives decision-grade figures.
- **Translation**: **Directions** (pick one or more of English → German, German → English, English → Spanish, Spanish → English; each is its own measurement), **Text size** (short messages or paragraphs), **Items in the quality pass** per direction.
- **Embeddings**: **Text size**, **Texts in the quality pass**.
- **Agent decisions**: **State size** (a short customer message, what an agent routes on first contact, or a ticket thread of 200 to 250 words with the deciding facts in the middle), **Questions per request** (1, 3 or both: the state is read once and each question adds only its own tokens, so three should cost little more than one; measuring both shows whether that holds on each machine), **Items in the quality pass**.
- **Intel's benchmark**: the input/output shape (128/128, 2,048/128, 128/2,048, 2,048/2,048 tokens).

The **Test data** page in the sidebar shows the same data in more depth at any time.

### 5.3 Step 3: Quick check or full test? (depth)

- **Quick check**: 50 requests for each measurement, run once. Shows the shape of the result in under an hour. The report marks every number as **indicative only (SMOKE)**; it must not drive a decision.
- **Full test**: 200 requests for each measurement, repeated 3 times. The minimum for trustworthy percentiles and for seeing whether a result repeats. This is what the report calls **decision-grade**.
- **Custom**: your own number of requests and repetitions. The report grades it as full only if it reaches 200 requests and 3 repetitions.

Why 200 and 3: a 99th-percentile latency ("99 of 100 requests within X") is the slowest 1 per cent; with 50 requests that is one request, which is noise. Repeating three times shows whether the result is stable.

### 5.4 Step 4: Which model?

One model per benchmark, so the hardware can be sized for it. Each card shows the model's name, what it does, its size in billions of parameters, its license, the number formats it can run in and which hardware options can serve it. A model whose license is research-only is marked; its results are for evaluation, not a commercial decision.

The shipped catalog:

| Model | Kind | Use cases |
|---|---|---|
| Qwen3 1.7B | chat | chat, detection, Intel's benchmark |
| Gemma 4 E2B / E4B Instruct | chat (Google's newest small models) | chat, detection, translation |
| RoBERTa-large (MNLI) | classifier | chat (classification), detection |
| EmbeddingGemma 300M | embedding | embeddings |
| Strands Decider E4B / 12B | decision model | agent decisions, detection |

More models can be added from Hugging Face under **Setup**. Models removed from the catalog stay visible in old reports.

### 5.5 Step 5: Which hardware to compare?

This is where the GPU and the CPU sides are chosen.

**Hardware options.** Tick the options to include. Each is a machine family plus the serving software on it: **GPU · NVIDIA · vLLM** and **CPU · Intel Xeon with AMX · vLLM**. Each ticked option has:

- **Instance size**: the AWS machine type, with its vCPUs, memory, GPU and hourly On-Demand price. The report prices results at the On-Demand rate of the size that ran.
- **Number formats**: BF16 is the like-for-like baseline on both sides. INT8 (W8A8) is a compressed format that is faster and cheaper on AMX; the report shows what it costs in answer quality. FP16 exists on the GPU only (the AWS Intel instances offered here lack AMX-FP16).

**Proposed pair for this test.** Press **Propose** and a reasoning model (Claude Opus on Amazon Bedrock, changeable under Setup) works out which GPU size and which CPU size make a fair comparison for this model and test:

1. It receives the model, the test's shape, every allowed size with its catalog specs and prices, and this tool's fairness rule.
2. It **searches the web** for what it is not sure of: the model's size and architecture, the GPU's memory, today's prices, whether the serving software supports the model. You watch the searches appear one by one; it usually takes one to three minutes.
3. If this model (or this test) has **already run on this tool**, the finished runs are then handed to it: which sizes ran, how busy each machine was, what the sizing verdict suggested, which sizes AWS refused for capacity. Measured runs outrank web claims when they disagree, and the model gives its **final verdict**. The proposal card then carries a line **What earlier runs here showed**, and says when the runs changed the sizes the web alone had suggested.
4. The result is two cards (GPU size, CPU size) with a plain-English **why** each, a **Why this is a fair comparison** note with the price ratio, a **confidence** badge, a **Good to know** list of caveats, and a collapsible **What was checked** with every fact, every search and every source URL. Press **Use this proposal** to apply it, or pick sizes by hand.

The **fairness rule**: the GPU's and the CPU's hourly prices must be within half to double of each other (closer is fairer), because the question is "for the same spend, what does each architecture deliver". A pair outside the band is flagged, and the review step asks you to confirm before starting.

**How to run the hardware options**: in parallel (fastest; one instance per option is up at once) or one after another (fewer instances at any moment; the run takes longer). Results are the same either way.

**How to buy the test instances**: **Spot** (spare capacity at a discount that AWS may reclaim with two minutes' notice; the run then launches a replacement and re-runs only the unfinished measurements) or **On-Demand** (not subject to Spot interruption, full price; capacity can still be refused, which the run reports). **Fall back to On-Demand** after N minutes without Spot capacity is ticked by default. Cost columns always use On-Demand prices so results stay comparable whatever you bought.

**Give up on a hardware option after N refusals**: AWS sometimes has no machines of a size in the zone, especially GPUs. The run asks again about once a minute, trying the alternate sizes too; after this many refusals it marks that option "Did not run, no capacity" and finishes the other options. The report then says so prominently, and the refusals themselves become evidence in the availability section of the results.

### 5.6 Step 6: How hard to push, and what counts as acceptable? (load and targets)

**Use typical tests** fills in the load settings and targets a first benchmark of this kind should use; the sentence next to it explains them. Everything stays editable.

**Request shape**:
- **Single request**: one at a time. Shows each machine's best-case latency.
- **Concurrent requests**: several in flight, as a service under load. Pick how many (each number is measured separately). Waiting time and throughput both rise with more concurrent requests; the verdict looks for the highest number where the machine still meets your targets.
- **Batched requests**: several inputs per request (classifiers, detectors, embedding models). Pick the batch sizes.

**What counts as acceptable** (your targets). A hardware option must meet these to count; the report marks every measurement that does, then picks the fastest GPU and the fastest CPU among those. Which targets you see depends on the use case:

| Target | Meaning | Typical value |
|---|---|---|
| **First word within (ms)** | The wait before the answer starts appearing, for 99 of 100 requests. | 2,000 ms for chat |
| **Time per generated word (ms)** | Median time between two generated tokens. | 150 ms is about reading speed; Intel uses 100 ms |
| **Whole answer / decision / request within (ms)** | The whole response, for 99 of 100 requests. | 1,000 ms for a real-time check, 2,000 ms for a guardrail in an agent's loop, 10 s for a paragraph translation |
| **Max CPU instances per GPU** | How many CPU machines you would accept in place of one GPU. | 4 |

For sensitive-data detection you pick a **preset** instead: Inline strict (every decision within 250 ms), Inline standard (1 second), Bulk scan (30 seconds, maximise documents per second). Empty means no limit.

### 5.7 Step 7: Review and start

- **Name this benchmark**: optional. It becomes the first part of the run id, followed by the start time.
- **What will run**: the whole test in words: use case, data, depth, model, hardware with sizes and formats, load settings, targets, purchase option.
- **Estimate**: wall-clock time, cost (On-Demand rates; Spot is usually 30 to 70 per cent less, see section 1.5) and the grade (Decision-grade or Quick check). The time is estimated from assumed speeds until the model has been measured here; afterwards it uses measured speeds.
- Problems found by the pre-checks are listed here in words: a size with too little memory for the model, a model the serving software cannot run in the chosen format, an unfair pair. An unfair pair can be started anyway by ticking **I understand the sizes are not comparable; start anyway**.
- **Start the benchmark** launches the run and takes you to its page under Runs.

---

## 6. Test data

What the models are asked during a run, per use case. Nothing is downloaded or generated at run time; every data set is prepared in advance with a fixed random seed so every run of the same test asks the same questions.

Pick a use case tile and you see, depending on the use case: the prompt sets with sample prompts (wrapped in each model's own chat template, so the exact text differs slightly per model while the content is the same); the detection scenarios with real example documents, their expected label and the sensitive values highlighted, including the **hard cases** (documents that look sensitive but are not: masked card numbers, documentation placeholders, public business data, first names alone); the parallel translation corpus; the decision items with their gold answers; Intel's prompt shape.

Why synthetic data: the detection dataset contains realistic-looking secrets and identifiers (card numbers, API keys, medical records). They are generated, so no real customer data is ever benchmarked, and the answers are known exactly, which is what makes accuracy measurable.

---

## 7. Runs

### 7.1 The list

Every benchmark, newest first, with its name, use case, model, grade (Decision-grade or Quick check), purchase option, status and when it last changed. Statuses:

- **Starting**: pre-checks running, instances launching.
- **Running**: measuring. Click the run for live progress.
- **Finished**: the report is under Results.
- **Finished — GPU (or CPU) side did not run**: the other side has numbers, this side was never served (no capacity). The report is one-sided and says so.
- **Refused by the pre-checks**: nothing started; the reason (missing capacity, price, data or access) is shown when you open the run.
- **Stopped by the operator**, **Failed**, **Timed out**: what they say.

### 7.2 One run

Open a run to see three tables that update live:

- **Instances**: one test instance per hardware option, with its state straight from EC2, the market (Spot, or On-Demand and whether it was a fallback), the size that actually launched (an alternate when the first was refused), capacity refusals and Spot interruptions.
- **Models on hardware options**: one row per model on one hardware option, going Waiting for the instance → Loading the model → Measuring → Done. "Queued behind X" means another model of this run is using the instance right now; "Not run, no capacity" means the option was given up.
- **Measurements**: one row per load setting, repetition, model and option, with requests done so far and errors. The accuracy pass of the detection use case classifies every test document once.

Buttons: **Stop** (running runs only; instances are terminated right away, finished measurements are kept), **Copy to new run** (prefills the wizard with this run's settings), **Delete** (finished runs only; removes the report, every measurement, the raw records and the logs; you are asked to confirm).

---

## 8. Results

This is the page that answers the question. Open it with one run selected and read it top to bottom; the order is deliberate.

### 8.1 Runs

Pick the run (or runs) to read. Type part of a run's name, model or date to find it. One run gives you its results; several put them side by side, for example the same model on two dates, or two models for the same job. Each selected run shows as a chip with its use case, data, request shape, concurrency, grade, model and instances.

### 8.2 Can these be compared?

Re-evaluated every time the selection changes.

- **Comparable**: every selected run answers the same question (same use case, same test data and answer length, same request shape), and every GPU and CPU instance pair sits within the fair price band.
- **Not comparable** (a blocker): the tool refuses to put the numbers side by side and says why in words. Different use cases, different prompt lengths, single versus concurrent requests: these load a machine differently and would compare apples with oranges.
- **Warnings**: differences to keep in mind (different concurrency lists that overlap only partly, a quick check among full tests).
- **One side did not run**: a yellow note. What follows is a one-sided picture, not a GPU-versus-CPU comparison.

### 8.3 Your targets

Prefilled with the targets the run was started with. Change them and every verdict below moves: that is the point. For instance, relax "first word within" from 2,000 to 4,000 ms and watch which CPU measurements start to qualify. **Max CPU instances per GPU** is your tolerance for replacing one GPU with several CPUs. **View** switches between all hardware options on the same charts and one series at a time.

### 8.4 Your load and your prices

The two numbers the decision really depends on. Nothing above this box knows them; everything below uses them.

- **Load to serve (requests per minute)**: how many requests your application sends at the level you want to size for (the peak hour, not the average). Chips give typical figures: **Pilot** 10 per minute, **Team** 60, **Service** 600, **Product** 6,000; or type your own. A request is one API call: one chat turn, one document, one batch of texts. Leave it empty and the verdicts assume you need a GPU's full peak.
- **Prices**: which hourly rates to use on both sides. **On-Demand** (what the measurements used), or a **Savings Plan**: Compute 1 year or 3 years, EC2 Instance 1 year or 3 years, no upfront payment. These are real published AWS rates; they lower both sides, and the CPU side a little more. **Your own rate** per hour overrides the basis for that side: enter **0** for capacity you already pay for.

### 8.5 Where the CPU is the better choice

The verdict, computed from the selected runs and never asserted. The headline sentence says whether the CPU has any benefit at all and, if so, which. Then up to four numbered sections:

**1 · Cost at your kind of load.** For every question the run answers (one per model and scenario), the load range in which the cheapest CPU option that meets your targets costs less than the cheapest GPU option. A small table gives worked examples at typical loads: at 10 requests per minute, the cheapest CPU option, its monthly cost, the cheapest GPU option, its monthly cost, and how many times cheaper. The **crossover** is the load above which the GPU's throughput per dollar wins; below it a small CPU slice beats the smallest GPU you can buy. A CPU size smaller than the one measured is **estimated** from core count and marked as such; one run on it confirms the figure. If no CPU option met the targets at any load, the question is marked "only the GPU met the targets".

**2 · Availability.** From the run's own capacity log: how often each side's launch was refused, by market (Spot or On-Demand). A GPU refused On-Demand means the type was unavailable; refused on Spot only is a mild edge. The badge says how real the benefit was in this run.

**3 · Smallest unit you can buy.** The cheapest instance that holds the model on each side, with its hourly price. A GPU costs its full price idle or busy; many small workloads can each have their own CPU slice.

**4 · Not measured here, but real.** Two structural points that no benchmark can measure but that belong in the decision: capacity you already pay for (enter 0 as your CPU rate above to see it), and same fleet, same tooling (no GPU driver stack, inference next to the application).

**Where the CPU is not the better choice** lists, per question, where the GPU wins and by how much; and the fixed statement that on peak throughput per dollar the GPU wins every question, so a workload that needs a GPU's full capacity should run on the GPU.

**How to read these results** below the box is a short paragraph of context, the same every time, so a reader who lands here cold is not misled by the headline.

### 8.6 Executive summary

Press **Write the executive summary** and the reasoning model (Claude Opus on Bedrock, changeable in Setup) writes a two-page summary of the findings below in plain English, with these sections: Bottom line, What was tested, Findings (a table with one row per question), Where the CPU is the better choice, At your load, Was the hardware the right size, Caveats, Recommendation, Suggested next step, Technical details. The model only sees the computed findings and is instructed to use no number that is not in them. It is regenerated when you change runs, targets, load or prices; each generation is one Bedrock call billed to the account, and the result is cached so re-opening the page is free. **Rewrite** forces a new one.

### 8.7 One card per question

Below the summary comes one card per **question**: one model with one scenario (for chat: the prompt set; for detection: one workload and document size; for decisions: one state size and questions-per-request; for embeddings: one text size and batch size). Each card has:

**Verdict.** One sentence: how many CPU instances match one GPU's peak throughput and at what cost ratio, or that no CPU (or GPU) measurement meets your targets, or that a side did not run.

**Two head-to-head tiles**, GPU green and CPU blue, each showing the **best** measurement on that side among those that meet your targets: instance, software, number format, the load setting it was reached at, hourly price with the Savings Plan rates underneath, and three big numbers:
- **Throughput**: work per second (tokens, documents, classifications, characters, embeddings or decisions). Higher is better. This is the capacity figure.
- **Responsiveness**: the time within which 99 of 100 requests completed (or the first word arrived, for chat). Lower is better.
- **Cost per unit**: the hourly price divided by the throughput, as if the machine were held at that load for an hour. The unit matches the use case: per million tokens, per GB scanned, per million decisions.

Under the numbers, the quality line for the use case: **Detection accuracy** (recall, its confidence interval, false positives, F1, agreement with the GPU, and a badge when it is below the accuracy floors), **Translation quality** (chrF), **Embedding quality** (cosine to the GPU's vectors, cross-lingual retrieval), **Decision quality** (accuracy on route, urgency and severity, mean confidence, agreement with the GPU), or **Answer quality loss vs GPU** for chat (1 minus the ROUGE-L similarity to the GPU's answers). A **Quality flag** is a warning, not an exclusion.

**Why it came out this way.** A short paragraph computed from the run: how much each request reads and writes, the share of time spent writing, the memory bandwidth each side reached against what it has, the arithmetic rate against each side's peak. Reading (processing the prompt) is arithmetic, which AMX is built for; writing (generating tokens) re-reads the whole model from memory for every token, where a price-matched CPU slice is furthest behind a GPU. This is why the gap is small for classifiers and embeddings and large for long chat answers.

**At your load** (when you entered a load above). Each option's capacity is the highest request rate at which it still met your targets; the number of instances is your load divided by that capacity, rounded up; the monthly cost is that many instances at your chosen prices. The cheapest option that meets your targets is named.

**How many CPU instances replace one GPU?** A bar: the count of the CPU machine needed to match the best GPU's throughput, against your limit.

**As the number of requests at a time grows.** Three charts across the load settings: throughput, responsiveness (with your target as a dashed line) and cost per unit. Each hardware option and format is one line; the legend names them.

**Was the hardware the right size?** From the monitoring agent on each test host: how busy the GPU (or the whole CPU) was during each measurement. The verdict reads the heaviest load setting:
- **Bigger than this load needs**: mostly idle; a smaller size (named, with its price) would do, or the load was too light to find its limit.
- **A good match for this load**: working hard with headroom.
- **Fully used, still meeting the targets**: at its limit, but within your targets.
- **Too small for this load**: saturated and missing your targets; the next size up is named.

Each suggestion says whether it still pairs fairly with the other side. **Re-run with recommended sizes** opens the wizard prefilled with the suggested sizes (and tells you when the new pair leaves the fair band). CPU busy is the share of the physical cores: the serving software runs one thread per core, so the host's vCPU figure can never read above 50 per cent; 100 per cent here means every core was busy.

**How busy the hardware was**: the same, as a line per option across load settings, with the idle and saturated bands shaded.

**All measurements in this group**: a collapsible table of every row (option, format, load setting, throughput, latency percentiles, error rate, price, cost per unit, AMX verdict, quality).

### 8.8 Utilization over the run

CPU busy, host memory, GPU busy and GPU memory for each test machine across the whole run, read from CloudWatch. Shaded bands are the measurements, labelled with their load setting; the gaps between them are model loading, warm-up and the accuracy pass. This is where you see, for example, that a CPU option sat at 100 per cent for the whole run.

### 8.9 Notes from the reports

Every warning the run produced, verbatim: SMOKE grade, AMX verdicts that are not VERIFIED and why, sides that did not run, data or price caveats. Read them before quoting a number.

**Download report** at the bottom gives you the raw report (summary.md, summary.json, summary.csv and the charts) for your own slides or spreadsheet.

---

## 9. Reading the numbers correctly

### 9.1 Throughput, responsiveness, cost: which one matters

- If your problem is **"can it keep up with the traffic"**, read **throughput**, and the **At your load** card, which converts throughput into a number of machines for your load.
- If your problem is **"is it fast enough for a user who is waiting"**, read **responsiveness** against your target. A machine that is cheap but makes a person wait 20 seconds fails.
- If your problem is **"what will it cost"**, read **cost per unit** for the peak case and **At your load** for your case. They differ: cost per unit assumes the machine is full; your load may leave the GPU mostly idle, which is exactly when the CPU wins.

### 9.2 Percentiles

"99 of 100 requests within X ms" is the 99th percentile. It describes the unlucky user, not the average one. The cards show P1/P50/P99 (the fastest, the typical, the slow tail). Targets are set on P99 for a reason: a service is judged by its slow tail.

### 9.3 Grade

A **Quick check (SMOKE)** shows the shape of the result. It is right about the order of magnitude and wrong about the details; the slow tail in particular is noise with 50 requests. Use it to decide what to test fully, never to decide a purchase. **Decision-grade** means at least 200 requests per measurement and three repetitions.

### 9.4 AMX verified

For every CPU measurement the tool proves that the processor's matrix units were used, by two independent means: the hardware counter `exe.amx_busy` during the measurement, and the serving software's kernel trace from a probe request. **VERIFIED** means both agree. **INVALID** means the CPU result ran without AMX and is excluded from the rankings (it would understate the CPU). **NOT_APPLICABLE** appears for the decision models, whose server runs the CPU side in 32-bit floating point by its own design; AMX has no 32-bit path, so the CPU uses AVX-512 instead. The result stays in the rankings and the report says so.

### 9.5 Number formats and the "effective precision" column

A cell is configured with a number format (BF16, INT8). The report shows the format the machine **actually ran**: an FP16 cell that silently ran BF16, a decision model that ran fp32 on CPU. When they differ the table says so, so no result claims a precision it did not use.

### 9.6 Quality and accuracy

Speed is half the story. Every use case with known answers scores them:

- **Detection**: recall (share of sensitive documents caught) with a 95 per cent confidence interval, false-positive rate, precision, F1, and whether the option meets the **accuracy floors** (recall at least 90 per cent, false positives at most 5 per cent by default). Below the floors, the option is marked but not hidden.
- **Translation**: chrF, a character-level similarity to the reference translation (0 to 100). Read it relative to the other options: a paraphrase lowers it for every option alike.
- **Embeddings**: cosine similarity to the GPU's vector for the same text (1.0000 = identical), and cross-lingual retrieval (does a German text's nearest English neighbour turn out to be its own translation).
- **Decisions**: accuracy per question against gold answers, mean confidence, and agreement with the GPU's decisions.
- **Chat**: answer quality loss against the GPU's own answers on fixed prompts; BF16 on CPU sets the noise floor, compressed formats are judged against it.

The same model on two machines should score alike. A gap between the CPU and the GPU on the same model is numerical noise, not a different model; a gap between two models is a real difference.

### 9.7 Prices

All measured cost columns use **On-Demand** prices from the AWS Price List API, snapshotted with the catalog (the date is shown), so runs bought on Spot and runs bought On-Demand stay comparable. Savings Plan rates are shown next to every tile and can be applied to the whole page from **Your load and your prices**. What you actually paid for the benchmark run itself is a separate matter (shown in the estimate before you start).

### 9.8 Comparing two models

Select two runs with different models but the same use case and test data. The comparability check must pass. Then every card shows both models' options on the same charts and the executive summary covers both. This is how you answer "would a smaller model on a CPU do the job that a bigger one does on the GPU".

---

## 10. Setup

Operators only.

**Hugging Face token.** Some models are gated: the Gemma family (Gemma 4, EmbeddingGemma, the base of the Strands Deciders) requires that you accept its license on huggingface.co and download with an access token of that account; a *read* token is enough (Hugging Face → Settings → Access Tokens). Qwen3, RoBERTa and the other non-gated models need none. Paste the token here and press **Save token**; it is stored in AWS Secrets Manager (never shown again), read by the benchmark hosts when they download a gated model and by the model-adding job. The badge says **Set · last changed …** when a token is stored and **Not set — gated models cannot be downloaded** when the slot is empty. Your administrator may already have stored one at deployment time (the deploy script takes it as an optional input), in which case the badge reads *Set*; a run of a gated model without a working token is refused by the pre-checks with a sentence naming the model, nothing starts and nothing is billed.

**Models.** The catalog a new benchmark can pick from. Each row shows the name, Hugging Face repository and exact version (revision) every run uses, the kind (chat, classifier, detector, embedding, decision), the size, the license, the engines that can run it (with **proven in a run** or **not yet run**), and the use cases. **Add a model** takes a Hugging Face repository id: the tool reads the model's configuration, works out its kind and size, prepares its test prompts with the model's own tokenizer (a few minutes), and adds it to the catalog. Models that need text generation and are of an unsupported architecture are refused with the reason. **Remove** only takes a model off the list; its old reports stay readable.

**Reasoning model.** The Amazon Bedrock model that proposes hardware (with web search) and writes executive summaries. Listed: the Anthropic Claude and OpenAI models this account can call in this region. Default: the newest Claude Opus.


---

## 11. Technical details

For the engineer who wants to check the method, and for the reader who wants to know what "we measured it" means here.

- **Architecture**: the picture of the AWS pieces (CloudFront and Cognito in front, a Fargate web application, Step Functions and Lambdas orchestrating, one EC2 test host per hardware option in a private network, S3 and DynamoDB for state, Bedrock for reasoning). Download the `.drawio` file to edit it.
- **How a benchmark runs**: the eight steps every run goes through, with a plain sentence each and **Technical specifics** for the exact component names.
- **Hardware options and their containers**: exactly which serving software (container image, pinned by digest) runs on which machine with which settings, so a result can be reproduced byte for byte.
- **Model-specific serving flags**: the few vendor-recipe options some models need, applied on both sides so neither gets a setting the other lacks.
- **Images this project builds**, **What happens to a model before each engine can serve it**, and **The files, verbatim**: the build recipes and host scripts of this exact build.

---

## 12. Costs and safety

- **What a run costs you**: the instances for the duration of the run (shown as an estimate before you start; Spot is usually 30 to 70 per cent less than the On-Demand estimate, section 1.5), a few cents of storage and Lambda time, and one Bedrock call per hardware proposal and per executive summary.
- **Instances are per run** and terminated when the run ends or is stopped. A reaper also terminates anything the tool launched that outlives its run.
- **Nothing is public**: the test hosts have no public IP addresses; the application sits behind CloudFront with a web firewall and a sign-in; the hosts reach the internet only to download models.
- **No real customer data** is used: every data set is synthetic, with known answers.
- **Everything is pinned**: model versions, container images, prices with their snapshot date, the test data with its hash. Two runs of the same test months apart measure the same thing.

---

## 13. Frequently asked questions

**The GPU won on throughput in my run. Is the CPU useless for this model?**
Read **Where the CPU is the better choice** and **At your load**. A GPU that is 7 times faster but 90 per cent idle at your load is 7 times the capacity you are paying for and not using. The crossover tells you the load above which the GPU is the cheaper option; below it, the CPU is.

**Why does the report say the CPU "needs 15 instances per GPU" when I would never buy 15?**
That is the peak-capacity view, kept because it is the honest answer to "can CPUs replace this GPU at full tilt". The decision view is **At your load**. Set your load and your limit; the verdict adapts.

**My run says one side did not run.**
AWS had no capacity for that size in the zone after the configured number of refusals. The other side's numbers stand alone. Re-run later, pick an alternate size, or raise the refusal limit. The refusals themselves are counted as evidence in the availability section.

**A CPU row is marked INVALID.**
The processor's AMX counters show the matrix units were not used, so the measurement would understate the CPU. It is excluded from rankings. The notes say what the probe found; the usual cause is a serving-software setting, which the Technical page lists.

**Can I benchmark my own model?**
Yes, under Setup → Models → Add a model, with its Hugging Face repository id. The first run is the real test of whether the serving software loads it; the hardware proposal says so in its caveats until a run has finished.

**Why Spot by default?**
It is typically 30 to 70 per cent cheaper (AWS Spot Instance Advisor, section 1.5) and the tool recovers from interruptions by relaunching and re-running only the unfinished measurements. The numbers are priced at On-Demand rates either way, so Spot never changes a result.

**Why is the quick check not enough to decide?**
Fifty requests give a 99th percentile made of one request. The full test's 200 requests and three repetitions are what make the slow tail and the repeatability trustworthy.

**The executive summary and the cards disagree with a number I computed from the report.**
The summary only sees the computed findings and is told to use no other number; the cards compute from the report at your current targets and prices. If you changed targets, load or prices, press **Rewrite**. If a disagreement remains, the report's `summary.json` (Download report) is the source of truth; tell the team.

**How do I share a result?**
The Results URL carries the selected runs (`/results?runs=...`); anyone with access can open it. **Download report** gives the files for slides. The executive summary is Markdown and pastes cleanly into documents.

---

## 14. Glossary

- **AMX**: Advanced Matrix Extensions, Intel's built-in matrix-math unit in Xeon processors since 2023. What makes a CPU competitive for inference.
- **BF16 / FP16 / INT8 / FP32**: number formats. BF16 is the standard full-quality format on GPUs and AMX CPUs; INT8 is compressed (half the memory, faster, small quality loss); FP32 is 32-bit, the most precise and the slowest.
- **Cell**: one model on one hardware option in one number format; the unit a run is planned in.
- **Concurrency / requests at a time**: how many requests are in flight simultaneously. The load setting.
- **Crossover**: the load (requests per minute) above which the GPU becomes the cheaper option.
- **Decision-grade / Quick check (SMOKE)**: at least 200 requests per measurement and three repetitions, versus fewer.
- **Engine / serving software**: the program that loads the model and answers requests: vLLM on both sides; the Strands Decider server for decision models.
- **Fair pair**: a GPU and a CPU instance whose On-Demand hourly prices are within half to double of each other.
- **Instance / size**: an AWS machine type (`g6.xlarge`, `m8i.4xlarge`). The tool proposes a pair; you may change it.
- **Measurement / load setting**: one combination of model, hardware option, format, request shape and concurrency, run once.
- **On-Demand / Spot / Savings Plan**: AWS pricing. Pay by the hour; discounted spare capacity that may be reclaimed; committed-use discount for one or three years.
- **P50 / P99**: the median and the 99th percentile of response times.
- **Question**: one model with one scenario; the unit the results page gives a verdict for.
- **Recall / false-positive rate / F1**: for detection, the share of sensitive items caught, the share of harmless items wrongly flagged, and their harmonic summary.
- **TTFT / TPOT**: time to the first token (the first word) and time per output token (per generated word). The two chat latencies.
- **Throughput**: work per second: tokens, documents, classifications, characters, embeddings or decisions.
- **Utilization**: how busy a processor was, as a share of its capacity; read every 10 seconds on every test host.
- **Workload / scenario**: in the detection use case, one of the four DLP tasks (bulk scan binary, bulk scan category, inline DLP, database columns).
