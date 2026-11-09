"""Build the DSPM/DLP dataset (design 24.3, Requirement 22).

    .venv/bin/python -m tools.dspm.build_dataset --out data/dspm/v1 \
        --hf-token-secret amx-gpu-bench/hf-token

Writes dev.jsonl.gz, test.jsonl.gz, manifest.json and review_sample.md. The build fails
if any check in tools/dspm/checks.py fails. Same seed, same source revisions and same
tool versions give byte-identical files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import platform
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dspm import checks, families, generator, sources, tables  # noqa: E402
from tools.dspm.checks import PRIORITY, label_for, top_category  # noqa: E402
from tools.dspm.generator import Component, derive_seed  # noqa: E402
from tools.dspm.scan import findings  # noqa: E402

REF_MODEL = "m1-qwen2.5-3b"
DSPM_MODELS = ("m1-qwen2.5-3b", "m2-qwen3-4b-2507", "m3-roberta-large-mnli", "m4-gemma-2-2b",
               "m5-flan-t5-xl", "d1-privacy-filter", "d2-roblox-pii-v2")
SEPARATORS = {"S1": "\n\n", "S2": "\n\n", "email": "\n\n---\n\n", "log": "\n", "config": "\n\n",
              "memo": "\n\n", "letter": "\n\n", "chat": "\n"}
PASTE_WRAPPERS = (
    "Can you check this form for mistakes before I send it?\n\n",
    "Summarize this document in three bullet points:\n\n",
    "Extract the key fields from this as JSON:\n\n",
    "Translate this into German, keep the formatting:\n\n",
    "Is anything missing from this? Be brief.\n\n",
)


@dataclass
class BuildConfig:
    seed: int = 20261109
    test_size: int = 3000
    dev_size: int = 300
    mix: dict = field(default_factory=lambda: {"W1-short": 0.2333, "W1-long": 0.2333, "W3": 0.4, "W4": 0.1334})
    pos_share: float = 0.5
    hard_share: float = 0.36
    short: tuple[int, int] = (250, 400)
    long: tuple[int, int] = (1500, 1900)
    w3: tuple[int, int] = (64, 360)
    w4: tuple[int, int] = (100, 950)
    # Flan-T5 sees the short policy and the options (125 tokens, 148 for W2 categories)
    # around the content and was trained on inputs of at most 512 tokens.
    caps_short: dict = field(default_factory=lambda: {"m5-flan-t5-xl": 358, "m3-roberta-large-mnli": 470})
    caps_w3: dict = field(default_factory=lambda: {"m5-flan-t5-xl": 380, "m3-roberta-large-mnli": 470,
                                                   "d2-roblox-pii-v2": 470})
    min_pos_category_share: float = 0.05
    min_pos_workload_share: float = 0.05
    min_hard_share: float = 0.30
    dev_share_sources: float = 0.10
    table_rows: int = 15


SOURCE_MIX_W1 = {
    "GOV_ID": {"S1": 0.5, "S2": 0.25, "S3": 0.25},
    "PERSONAL": {"S1": 0.6, "S2": 0.15, "S3": 0.25},
    "FINANCIAL": {"S1": 0.4, "S2": 0.35, "S3": 0.25},
    "HEALTH": {"S1": 0.7, "S3": 0.3},
    "CREDENTIALS": {"S1": 0.15, "S2": 0.15, "S3": 0.7},
    "CONFIDENTIAL": {"S3": 1.0},
}
W3_PASTE_SHARE = {"GOV_ID": 0.15, "PERSONAL": 0.15, "FINANCIAL": 0.15, "HEALTH": 0.15, "CREDENTIALS": 0.1}
HARD_POSITIVE_SHARE = {("W1", "CREDENTIALS"): ("test-labeled-secret", 0.2),
                       ("W3", "CREDENTIALS"): ("test-labeled-secret", 0.2),
                       ("W3", "GOV_ID"): ("keyword-proximity", 0.25)}


# ---------------------------------------------------------------------------- tokens

class TokenCounter:
    """Token counts per model. `fake` uses len(text)/4 for every model (tests only)."""

    def __init__(self, tokenizers: dict | None = None):
        self.toks = tokenizers
        self.cache: dict[tuple[str, str], int] = {}

    def count(self, model: str, text: str) -> int:
        key = (model, text)
        if key not in self.cache:
            if self.toks is None:
                self.cache[key] = max(1, math.ceil(len(text) / 4))
            else:
                self.cache[key] = len(self.toks[model](text, add_special_tokens=True)["input_ids"])
        return self.cache[key]

    def all(self, text: str) -> dict[str, int]:
        return {m: self.count(m, text) for m in DSPM_MODELS}


def load_tokenizers(token: str | None) -> dict:
    from bench.config import load_models
    from bench.prompts import tokenizer_for

    cat = load_models(ROOT / "config/models.yaml")
    by_id = {m.id: m for m in cat.models}
    return {mid: tokenizer_for(by_id[mid], token) for mid in DSPM_MODELS}


# ---------------------------------------------------------------------------- source pools

class Pools:
    """Per-split pools of S1/S2 records, consumed once each, in a seeded order."""

    def __init__(self, recs: list[sources.Record], smap: sources.SpanMap, seed: int, dev_share: float):
        self.smap = smap
        self.cats: dict[str, set[str]] = {}
        self.trusted: dict[str, bool] = {}
        self.order: dict[tuple[str, str], list[sources.Record]] = defaultdict(list)
        for r in recs:
            self.cats[r.uid] = smap.categories(sources.SOURCE_KEY[r.source], r.text, r.spans)
            self.trusted[r.uid] = sources.trusted(r)
            self.order[(sources.split_of(r.uid, dev_share), r.source)].append(r)
        for k, v in self.order.items():
            v.sort(key=lambda r: r.uid)
            random.Random(derive_seed(seed, "pool", *k)).shuffle(v)
        self.cursor: dict[tuple, int] = defaultdict(int)
        self.used: set[str] = set()

    def take(self, split: str, source: str, want: str | None = None, *, need_cats: bool | None = None,
             max_ref: int | None = None, counter: TokenCounter | None = None) -> sources.Record | None:
        lst = self.order.get((split, source), [])
        key = (split, source, want, need_cats, max_ref)
        i = self.cursor[key]
        while i < len(lst):
            r = lst[i]
            i += 1
            if r.uid in self.used:
                continue
            c = self.cats[r.uid]
            if want is not None and want not in c:
                continue
            if (want is not None or need_cats) and not self.trusted[r.uid]:
                continue
            if need_cats is True and not c or need_cats is False and c:
                continue
            if max_ref is not None and counter is not None and counter.count(REF_MODEL, r.text) > max_ref:
                continue
            self.cursor[key] = i
            self.used.add(r.uid)
            return r
        self.cursor[key] = i
        return None


def _s_positive(pools: Pools, r: sources.Record, rng: random.Random) -> Component | None:
    p = sources.positive(r, rng)
    key = sources.SOURCE_KEY[r.source]
    ev = pools.smap.evidence(key, p.text, p.spans)
    cats = sorted(pools.smap.categories(key, p.text, p.spans))
    for _kind, m in findings(p.text):
        pos = p.text.find(m)
        if not any(s["start"] < pos + len(m) and pos < s["end"] for s in ev):
            return None
    if not cats:
        return None
    return Component(p.text, ev, f"{r.source.lower()}/{r.doc_type}", "document", r.source, cats, uid=r.uid)


def _s_sanitized(pools: Pools, r: sources.Record, rng: random.Random) -> Component | None:
    s = sources.sanitize(r, rng, pools.smap)
    if s is None:
        return None
    return Component(s.text, [], f"{r.source.lower()}/{r.doc_type}", "document", r.source, [], uid=r.uid)


def _s_masked(pools: Pools, r: sources.Record, rng: random.Random) -> Component | None:
    out = sources.mask(r, rng, pools.smap)
    if out is None:
        return None
    m, originals = out
    return Component(m.text, [], f"{r.source.lower()}/{r.doc_type}", "document", r.source, [], "masked",
                     uid=r.uid, masked_originals=originals)


# ---------------------------------------------------------------------------- builder

def _alloc(n: int, keys: list[str]) -> dict[str, int]:
    base, rem = divmod(n, len(keys))
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}


def _pick(rng: random.Random, weights: dict[str, float]) -> list[str]:
    """Sources in a seeded order of preference: the weighted draw first, then the rest."""
    keys = sorted(weights)
    first = rng.choices(keys, [weights[k] for k in keys])[0]
    return [first] + [k for k in keys if k != first]


_EXT = {"code-py": "py", "code-js": "js", "code-go": "go", "code-java": "java", "code-tf": "tf"}
_WORDS = ["client", "handlers", "utils", "config", "models", "service", "main", "billing", "auth",
          "jobs", "routes", "settings", "storage", "metrics"]


def _with_path(c: Component, seed: int) -> Component:
    """Prefix a component of a repository-dump chunk with a file header (==> path <==)."""
    rng = random.Random(seed)
    w = rng.choice(_WORDS)
    if c.channel in _EXT:
        top = {"code-py": "src", "code-js": "web/src", "code-go": "internal", "code-java": "src/main/java",
               "code-tf": "infra"}[c.channel]
        path = f"{top}/{rng.choice(generator.SERVICES)}/{w}.{_EXT[c.channel]}"
    elif c.channel == "config":
        path = f"config/{rng.choice(generator.SERVICES)}-{w}.{rng.choice(['yaml', 'env', 'json', 'toml'])}"
    else:
        slug = "-".join(c.family.split("/", 1)[-1].lower().split())[:40] or "notes"
        path = f"docs/{slug}.md"
    head = f"==> {path} <==\n"
    spans = [{**s, "start": s["start"] + len(head), "end": s["end"] + len(head)} for s in c.spans]
    return Component(head + c.text, spans, c.family, c.channel, c.source, c.categories, c.hard_case,
                     uid=c.uid, template=c.template, masked_originals=c.masked_originals)


class Builder:
    def __init__(self, cfg: BuildConfig, s1: list[sources.Record], s2: list[sources.Record],
                 counter: TokenCounter, smap: sources.SpanMap):
        families.validate()
        self.cfg, self.counter, self.smap = cfg, counter, smap
        self.pools = Pools(s1 + s2, smap, cfg.seed, cfg.dev_share_sources)
        self.fams = families.FAMILIES
        self.seq: Counter = Counter()
        self.masked_originals: dict[str, list[str]] = {}
        self.seen_text: set[str] = set()

    # -- S3 component helpers
    def _families(self, channels, workload, *, kind, want=None, hard=None, filler=None, split="test"):
        out = []
        for f in self.fams:
            if f.channel not in channels or workload not in f.workloads or f.kind != kind:
                continue
            if hard is not None and f.hard_case != hard:
                continue
            if hard is None and kind == "neg" and f.hard_case:
                continue
            if kind == "pos" and hard is None and f.hard_case:
                continue
            if filler is not None and f.filler != filler:
                continue
            if generator.templates_for(f, split, want):
                out.append(f)
        return out

    def _gen(self, fams, split, seed, want=None) -> Component:
        rng = random.Random(seed)
        f = rng.choice(fams)
        return generator.generate(f, split, derive_seed(seed, "g"), want)

    def _filler(self, channel: str, split: str, seed: int) -> Component | None:
        # Half of the fillers of an S3 chunk come from a neighbouring source, so long chunks do
        # not repeat a few templates: code and configuration chunks mix in other code and
        # configuration files (a repository dump), email, memo and letter chunks mix in
        # sanitized S1 documents (a folder export). Logs stay logs.
        mix = random.Random(derive_seed(seed, "mix"))
        r = mix.random()
        if channel not in ("S1", "S2", "log", "chat") and r < 0.5:
            if (channel.startswith("code-") or channel == "config") and r >= 0.25:
                other = mix.choice([c for c in (*families.CODE_CHANNELS, "config") if c != channel])
                fams = [f for f in self.fams if f.channel == other and f.filler]
                return self._gen(fams, split, derive_seed(seed, "code"))
            doc = self._filler("S1", split, derive_seed(seed, "doc"))
            if doc is not None:
                return doc
        if channel in ("S1", "S2"):
            rng = random.Random(seed)
            for _ in range(20):
                r = self.pools.take(split, channel, need_cats=None, max_ref=self.cfg.long[1] // 3,
                                    counter=self.counter)
                if r is None:
                    return None
                c = _s_sanitized(self.pools, r, rng)
                if c is not None:
                    return c
            return None
        fams = [f for f in self.fams if f.channel == channel and f.filler]
        return self._gen(fams, split, seed)

    # -- assembly
    def _join(self, parts: list[Component], sep: str, prefix: str = "") -> tuple[str, list[dict]]:
        text, spans = prefix, []
        for i, c in enumerate(parts):
            if i:
                text += sep
            off = len(text)
            spans += [{**s, "start": s["start"] + off, "end": s["end"] + off} for s in c.spans]
            text += c.text
        return text, spans

    def _fits(self, text: str, lo: int, hi: int, caps: dict) -> bool:
        n = self.counter.count(REF_MODEL, text)
        return lo <= n <= hi and all(self.counter.count(m, text) <= cap for m, cap in caps.items())

    def _chunk(self, primary: Component | None, channel: str, split: str, seed: int,
               lo: int, hi: int, caps: dict) -> tuple[str, list[dict], list[Component]] | None:
        rng = random.Random(seed)
        sep = SEPARATORS.get(channel, "\n\n")
        parts = [primary] if primary else []
        fillers: list[Component] = []
        for k in range(60):
            text, _ = self._join(parts + fillers, sep)
            n = self.counter.count(REF_MODEL, text) if text else 0
            if n >= lo:
                break
            f = self._filler(channel, split, derive_seed(seed, "f", k))
            if f is None:
                return None
            trial, _ = self._join(parts + fillers + [f], sep)
            if self.counter.count(REF_MODEL, trial) <= hi:
                fillers.append(f)
        rng.shuffle(fillers)
        order = list(fillers)
        if primary:
            order.insert(rng.randrange(len(order) + 1), primary)
        if not order:
            return None
        if channel.startswith("code-") or channel == "config":
            order = [_with_path(c, derive_seed(seed, "path", i)) for i, c in enumerate(order)]
        text, spans = self._join(order, sep)
        if not self._fits(text, lo, hi, caps):
            return None
        return text, spans, order

    def _item(self, split, workload, bucket, text, spans, comps, primary) -> dict | None:
        norm = " ".join(text.lower().split())
        if norm in self.seen_text:
            return None
        cats = sorted({s["category"] for s in spans}, key=PRIORITY.index)
        src = primary.source if primary else comps[0].source
        channel = primary.channel if primary else comps[0].channel
        self.seq[split] += 1
        bucket_part = f"-{bucket}" if bucket else ""
        item = {
            "id": f"{split}-{workload.lower()}{bucket_part}-{src.lower()}-{channel}-{self.seq[split]:05d}",
            "split": split, "workload": workload,
            "workloads": [workload, "W2"] if (workload == "W1" and bucket == "short") else [workload],
            "bucket": bucket, "channel": channel, "source": src,
            "family": primary.family if primary else "filler-only",
            "text": text, "sensitive": bool(cats), "categories": cats,
            "category_top": top_category(cats), "label": label_for(workload, cats),
            "hard_case": primary.hard_case if primary else None,
            "spans": sorted(spans, key=lambda s: (s["start"], s["end"])),
            "components": [{"source": c.source, "family": c.family, "uid": c.uid, "template": c.template}
                           for c in comps],
            "tokens": self.counter.all(text),
            "bytes": len(text.encode("utf-8")),
        }
        originals = [o for c in comps for o in c.masked_originals]
        errs = checks.check_item(item, originals)
        if errs:
            return None
        self.seen_text.add(norm)
        if originals:
            self.masked_originals[item["id"]] = originals
        return item

    # -- W1
    def _w1_slot(self, split, bucket, polarity, tag, hard_pos, seed) -> dict | None:
        cfg = self.cfg
        lo, hi = cfg.short if bucket == "short" else cfg.long
        caps = cfg.caps_short if bucket == "short" else {}
        max_ref = hi if bucket == "short" else hi // 2
        rng = random.Random(seed)
        for attempt in range(40):
            s = derive_seed(seed, attempt)
            primary, channel = None, None
            if polarity == "pos":
                mix = SOURCE_MIX_W1[tag] if not hard_pos else {"S3": 1.0}
                for src in _pick(random.Random(s), mix):
                    if src == "S3":
                        fams = self._families(families.W1_S3_CHANNELS, "W1", kind="pos", want=tag,
                                              hard=hard_pos, split=split)
                        if fams:
                            primary = self._gen(fams, split, s, want=tag)
                            channel = primary.channel
                    else:
                        r = self.pools.take(split, src, want=tag, max_ref=max_ref, counter=self.counter)
                        if r is not None:
                            primary = _s_positive(self.pools, r, rng)
                            channel = src
                    if primary is not None:
                        break
            elif tag == "easy":
                src = _pick(random.Random(s), {"S1": 0.4, "S2": 0.2, "S3": 0.4})[0]
                if src == "S3":
                    channel = random.Random(s).choice(families.W1_S3_CHANNELS)
                else:
                    r = self.pools.take(split, src, max_ref=max_ref, counter=self.counter)
                    primary = _s_sanitized(self.pools, r, rng) if r is not None else None
                    channel = src
                    if primary is None:
                        continue
            elif tag == "masked" and random.Random(s).random() < 0.5:
                src = _pick(random.Random(s), {"S1": 0.7, "S2": 0.3})[0]
                r = self.pools.take(split, src, need_cats=True, max_ref=max_ref, counter=self.counter)
                primary = _s_masked(self.pools, r, rng) if r is not None else None
                channel = src
            else:
                fams = self._families(families.W1_S3_CHANNELS, "W1", kind="neg", hard=tag, split=split)
                if fams:
                    primary = self._gen(fams, split, s)
                    channel = primary.channel
            if channel is None or (primary is None and not (polarity == "neg" and tag == "easy")):
                continue
            got = self._chunk(primary, channel, split, derive_seed(s, "chunk"), lo, hi, caps)
            if got is None:
                continue
            text, spans, comps = got
            item = self._item(split, "W1", bucket, text, spans, comps, primary)
            if item is not None and (polarity == "neg") == (not item["sensitive"]):
                return item
        return None

    # -- W3
    def _paste(self, split, r_comp: Component, seed) -> Component:
        rng = random.Random(seed)
        choices = [i for i in range(len(PASTE_WRAPPERS)) if generator.template_split(i) == split]
        wi = rng.choice(choices)
        pre = PASTE_WRAPPERS[wi]
        spans = [{**s, "start": s["start"] + len(pre), "end": s["end"] + len(pre)} for s in r_comp.spans]
        return Component(pre + r_comp.text, spans, "genai/paste-document", "genai", r_comp.source,
                         r_comp.categories, r_comp.hard_case, uid=r_comp.uid, template=wi,
                         masked_originals=r_comp.masked_originals)

    def _thread(self, msg: Component, split: str, seed: int, lo: int) -> tuple[str, list[dict], list[Component]]:
        """A chat thread: the message plus benign lines until the thread has at least lo tokens."""
        rng = random.Random(seed)
        names = [generator.Ctx(derive_seed(seed, "n", k)).get("first_name") for k in range(3)]
        others: list[Component] = []
        target = rng.randint(1, 4)
        while True:
            order = others[:]
            order.insert(derive_seed(seed, "pos") % (len(order) + 1), msg)
            text, spans = "", []
            for i, c in enumerate(order):
                pre = f"{names[i % 3]}: "
                off = len(text) + len(pre)
                spans += [{**sp, "start": sp["start"] + off, "end": sp["end"] + off} for sp in c.spans]
                text += pre + c.text + ("\n" if i < len(order) - 1 else "")
            if (len(others) >= target and self.counter.count(REF_MODEL, text) >= lo) or len(others) >= 12:
                return text, spans, order
            used = {(o.family, o.template) for o in others}
            for k in range(10):
                f = self._filler("chat", split, derive_seed(seed, len(others), k))
                if (f.family, f.template) not in used:
                    break
            others.append(f)

    def _pad_w3(self, comp: Component, split: str, seed: int, lo: int) -> tuple[str, list[dict], list[Component]] | None:
        """Bring a W3 item to at least lo tokens with benign context from the same channel:
        quoted earlier emails, a context document before a GenAI prompt, the rest of the file
        after a code question. Chat messages become threads."""
        rng = random.Random(seed)
        if comp.channel == "chat":
            if rng.random() < 0.5 or self.counter.count(REF_MODEL, comp.text) < lo:
                return self._thread(comp, split, derive_seed(seed, "t"), lo)
            return comp.text, comp.spans, [comp]
        text, spans, comps = comp.text, list(comp.spans), [comp]
        for k in range(8):
            if self.counter.count(REF_MODEL, text) >= lo:
                return text, spans, comps
            if comp.channel == "email":
                f = self._filler("email", split, derive_seed(seed, "e", k))
                if f is None or f.channel != "email":
                    continue
                text += "\n\n-----Original Message-----\n" + f.text
            elif comp.channel == "code_question":
                m = re.search(r"```(\w+)", comp.text)
                lang = {"python": "code-py", "js": "code-js", "go": "code-go", "java": "code-java",
                        "hcl": "code-tf", "yaml": "config"}.get(m.group(1) if m else "", "code-py")
                fams = [x for x in self.fams if x.channel == lang and x.filler]
                f = self._gen(fams, split, derive_seed(seed, "c", k))
                if f.text in text:
                    continue
                block = "\n\nRest of the file:\n```" + (m.group(1) if m else "") + "\n"
                if "Rest of the file:" in text:
                    text = text[:-4] + "\n\n" + f.text + "\n```"
                else:
                    text += block + f.text + "\n```"
            elif comp.channel == "genai":
                f = self._filler("S1", split, derive_seed(seed, "d", k)) if rng.random() < 0.7 else \
                    self._gen([x for x in self.fams if x.channel == "memo" and x.filler], split, derive_seed(seed, "m", k))
                if f is None:
                    continue
                pre = "Use this as context:\n\n" + f.text + "\n\nTask: "
                spans = [{**sp, "start": sp["start"] + len(pre), "end": sp["end"] + len(pre)} for sp in spans]
                text = pre + text
            else:
                return None
            comps.append(f)
        return (text, spans, comps) if self.counter.count(REF_MODEL, text) >= lo else None

    def _w3_slot(self, split, polarity, tag, hard_pos, seed) -> dict | None:
        cfg = self.cfg
        lo, hi = cfg.w3
        rng = random.Random(seed)
        for attempt in range(40):
            s = derive_seed(seed, attempt)
            r2 = random.Random(s)
            comp = None
            paste_share = W3_PASTE_SHARE.get(tag, 0) if polarity == "pos" else (0.15 if tag == "easy" else 0.3 if tag == "masked" else 0)
            if not hard_pos and r2.random() < paste_share:
                src = "S1"
                if polarity == "pos":
                    r = self.pools.take(split, src, want=tag, max_ref=hi - 40, counter=self.counter)
                    c = _s_positive(self.pools, r, rng) if r is not None else None
                elif tag == "easy":
                    r = self.pools.take(split, src, max_ref=hi - 40, counter=self.counter)
                    c = _s_sanitized(self.pools, r, rng) if r is not None else None
                else:
                    r = self.pools.take(split, src, need_cats=True, max_ref=hi - 40, counter=self.counter)
                    c = _s_masked(self.pools, r, rng) if r is not None else None
                comp = self._paste(split, c, s) if c is not None else None
            else:
                if polarity == "pos":
                    fams = self._families(families.W3_CHANNELS, "W3", kind="pos", want=tag, hard=hard_pos, split=split)
                    comp = self._gen(fams, split, s, want=tag) if fams else None
                elif tag == "easy":
                    fams = self._families(families.W3_CHANNELS, "W3", kind="neg", split=split)
                    comp = self._gen(fams, split, s) if fams else None
                else:
                    fams = self._families(families.W3_CHANNELS, "W3", kind="neg", hard=tag, split=split)
                    comp = self._gen(fams, split, s) if fams else None
            if comp is None:
                continue
            got = self._pad_w3(comp, split, derive_seed(s, "pad"), lo)
            if got is None:
                continue
            text, spans, comps = got
            if not self._fits(text, lo, hi, cfg.caps_w3):
                continue
            item = self._item(split, "W3", None, text, spans, comps, comp)
            if item is not None and (polarity == "neg") == (not item["sensitive"]):
                return item
        return None

    # -- W4
    def _w4_slot(self, split, polarity, tag, seed) -> dict | None:
        lo, hi = self.cfg.w4
        for attempt in range(40):
            s = derive_seed(seed, attempt)
            r2 = random.Random(s)
            if polarity == "pos":
                kinds = [k for k in tables.positive_types() if tag in tables.categories_of(k)]
            elif tag == "easy":
                kinds = tables.easy_negative_types()
            else:
                kinds = [k for k, h in tables.hard_types().items() if h == tag]
            comp = tables.generate(r2.choice(kinds), split, s, rows=self.cfg.table_rows)
            if not self._fits(comp.text, lo, hi, {}):
                continue
            item = self._item(split, "W4", None, comp.text, comp.spans, [comp], comp)
            if item is not None:
                return item
        return None

    # -- plan
    def plan(self, split: str) -> list[tuple]:
        cfg = self.cfg
        n_total = cfg.test_size if split == "test" else cfg.dev_size
        slots = []
        # Largest-remainder rounding, so the buckets add up to exactly n_total.
        raw = {wb: n_total * share / sum(cfg.mix.values()) for wb, share in cfg.mix.items()}
        sizes = {wb: math.floor(v) for wb, v in raw.items()}
        for wb in sorted(raw, key=lambda k: (sizes[k] - raw[k], k))[:n_total - sum(sizes.values())]:
            sizes[wb] += 1
        for wb in cfg.mix:
            n = sizes[wb]
            n_pos = round(n * cfg.pos_share)
            n_neg = n - n_pos
            wl, _, bucket = wb.partition("-")
            types = list(checks.HARD_NEGATIVE_TYPES[wl])
            n_hard = min(n_neg, max(math.ceil(n_neg * cfg.hard_share), len(types)))
            for cat, k in _alloc(n_pos, list(PRIORITY)).items():
                hp = HARD_POSITIVE_SHARE.get((wl, cat))
                n_hp = round(k * hp[1]) if hp else 0
                slots += [(wl, bucket or None, "pos", cat, hp[0] if i < n_hp else None) for i in range(k)]
            for t, k in _alloc(n_hard, types).items():
                slots += [(wl, bucket or None, "neg", t, None)] * k
            slots += [(wl, bucket or None, "neg", "easy", None)] * (n_neg - n_hard)
        return slots

    def build(self) -> tuple[dict[str, list[dict]], list[str]]:
        out: dict[str, list[dict]] = {}
        failures: list[str] = []
        for split in ("dev", "test"):
            items = []
            for k, (wl, bucket, pol, tag, hp) in enumerate(self.plan(split)):
                seed = derive_seed(self.cfg.seed, split, k, wl, bucket, pol, tag, hp)
                if wl == "W1":
                    it = self._w1_slot(split, bucket, pol, tag, hp, seed)
                elif wl == "W3":
                    it = self._w3_slot(split, pol, tag, hp, seed)
                else:
                    it = self._w4_slot(split, pol, tag, seed)
                if it is None:
                    failures.append(f"{split}: could not fill slot {wl}/{bucket}/{pol}/{tag}/{hp}")
                else:
                    items.append(it)
            out[split] = items
        return out, failures


# ---------------------------------------------------------------------------- outputs

def _canon(item: dict) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def dataset_hash(splits: dict[str, list[dict]]) -> str:
    h = hashlib.sha256()
    for split in ("dev", "test"):
        for it in splits[split]:
            h.update(_canon(it).encode())
            h.update(b"\n")
    return h.hexdigest()


def write_jsonl_gz(path: Path, items: list[dict]) -> str:
    raw = "".join(_canon(it) + "\n" for it in items).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(raw)
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _stats(vals: list[int]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {"min": s[0], "p50": s[len(s) // 2], "max": s[-1], "mean": round(sum(s) / len(s), 1)}


def manifest(cfg: BuildConfig, splits: dict, report: dict, file_hashes: dict, policy: dict,
             tokenizers_real: bool) -> dict:
    import faker
    import transformers

    counts = {}
    for split, items in splits.items():
        c = defaultdict(Counter)
        for it in items:
            c["workload"][it["workload"]] += 1
            c["label"][f"{it['workload']}:{it['label']}"] += 1
            c["source"][it["source"]] += 1
            c["channel"][it["channel"]] += 1
            if it["hard_case"]:
                c["hard_case"][f"{it['workload']}:{it['hard_case']}"] += 1
            for cat in it["categories"]:
                c["category"][f"{it['workload']}:{cat}"] += 1
        counts[split] = {k: dict(sorted(v.items())) for k, v in c.items()} | {"items": len(items)}
    tokens = {}
    for wb in ("W1-short", "W1-long", "W3", "W4"):
        wl, _, b = wb.partition("-")
        its = [it for it in splits["test"] if it["workload"] == wl and (it["bucket"] or "") == b]
        tokens[wb] = {m: _stats([it["tokens"][m] for it in its]) for m in DSPM_MODELS}
        tokens[wb]["bytes"] = _stats([it["bytes"] for it in its])
    return {
        "dataset": "dspm", "version": "v1", "dataset_sha256": dataset_hash(splits), "files": file_hashes,
        "seed": cfg.seed, "config": asdict(cfg), "counts": counts, "token_stats_test": tokens,
        "checks": report, "policy": policy,
        "sources": {
            "S1": {"repo": sources.NEMOTRON_REPO, "revision": sources.NEMOTRON_REV, "file": sources.NEMOTRON_FILE,
                   "license": "CC BY 4.0", "attribution": "NVIDIA Nemotron-PII (CC BY 4.0)"},
            "S2": {"repo": sources.GRETEL_REPO, "revision": sources.GRETEL_REV, "file": sources.GRETEL_FILE,
                   "license": "Apache-2.0"},
            "S3": {"generator_commit": _git_commit(), "families": len(families.FAMILIES),
                   "templates": sum(len(f.templates) for f in families.FAMILIES)},
        },
        "span_label_map_sha256": hashlib.sha256((ROOT / "config/dspm/span_label_map.yaml").read_bytes()).hexdigest(),
        "tools": {"python": platform.python_version(), "faker": faker.VERSION,
                  "transformers": transformers.__version__, "tokenizers": "real" if tokenizers_real else "fake"},
    }


def _mark(text: str, spans: list[dict]) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["start"] < pos:
            continue
        out += [text[pos:s["start"]], "⟦", text[s["start"]:s["end"]], f"⟧{{{s['category']}}}"]
        pos = s["end"]
    out.append(text[pos:])
    return "".join(out)


def review_sample(items: list[dict], n: int = 100, seed: int = 0) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        key = (it["workload"], it["bucket"] or "", it["sensitive"], it["hard_case"] or it["category_top"])
        groups[key].append(it)
    rng = random.Random(seed)
    for v in groups.values():
        rng.shuffle(v)
    keys = sorted(groups, key=str)
    chosen: list[dict] = []
    while len(chosen) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(chosen) < n:
                chosen.append(groups[k].pop())
    chosen.sort(key=lambda it: it["id"])
    lines = ["# DSPM/DLP dataset v1: review sample", "",
             f"{len(chosen)} test items stratified by workload, bucket, label and hard case or category. "
             "Sensitive values are marked ⟦like this⟧{CATEGORY}. Long items show the part around the first "
             "marked value. Tick the box when the label is right under config/dspm/policy_v1.md; note "
             "anything else next to it.", "",
             "All values are synthetic. Nemotron-PII items are CC BY 4.0 (NVIDIA).", ""]
    for it in chosen:
        text = _mark(it["text"], it["spans"])
        if len(text) > 1600:
            first = text.find("⟦")
            start = max(0, (first if first >= 0 else 0) - 500)
            text = ("[...]\n" if start else "") + text[start:start + 1400] + "\n[...]"
        lines += [f"## {it['id']}", "",
                  f"- [ ] label **{it['label']}**, categories {it['categories'] or '[]'}, hard case "
                  f"{it['hard_case'] or 'none'}",
                  f"- workload {it['workload']}{(' ' + it['bucket']) if it['bucket'] else ''}, source "
                  f"{it['source']}, family `{it['family']}`, {it['tokens'][REF_MODEL]} tokens ({REF_MODEL}), "
                  f"{it['bytes']} bytes", "", "````text", text, "````", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/dspm/v1")
    ap.add_argument("--cache", default="data/cache/hf")
    ap.add_argument("--seed", type=int, default=BuildConfig.seed)
    ap.add_argument("--test-size", type=int, default=BuildConfig.test_size)
    ap.add_argument("--dev-size", type=int, default=BuildConfig.dev_size)
    ap.add_argument("--hf-token-secret", default=None)
    ap.add_argument("--nemotron-limit", type=int, default=40000)
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download

    from bench.dspm import policy_hashes

    token = None
    if args.hf_token_secret:
        import boto3

        token = boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(
            SecretId=args.hf_token_secret)["SecretString"]
    cache = ROOT / args.cache
    smap = sources.SpanMap(ROOT / "config/dspm/span_label_map.yaml")
    p1 = Path(hf_hub_download(sources.NEMOTRON_REPO, sources.NEMOTRON_FILE, repo_type="dataset",
                              revision=sources.NEMOTRON_REV, cache_dir=cache))
    p2 = Path(hf_hub_download(sources.GRETEL_REPO, sources.GRETEL_FILE, repo_type="dataset",
                              revision=sources.GRETEL_REV, cache_dir=cache))
    s1 = sources.load_nemotron(p1, smap, limit=args.nemotron_limit, seed=args.seed)
    s2 = sources.load_gretel(p2, smap)
    print(f"sources: S1 {len(s1)} records, S2 {len(s2)} records", file=sys.stderr)
    counter = TokenCounter(load_tokenizers(token))
    cfg = BuildConfig(seed=args.seed, test_size=args.test_size, dev_size=args.dev_size)
    splits, failures, report = build(cfg, s1, s2, counter, smap)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if failures or report["errors"]:
        (out / "build_errors.json").write_text(json.dumps({"failures": failures, "report": report}, indent=2))
        print(f"BUILD FAILED: {len(failures)} unfilled slots, {len(report['errors'])} check errors; "
              f"see {out / 'build_errors.json'}", file=sys.stderr)
        for e in (failures + report["errors"])[:20]:
            print("  " + e, file=sys.stderr)
        return 1
    hashes = {f"{s}.jsonl.gz": write_jsonl_gz(out / f"{s}.jsonl.gz", splits[s]) for s in ("dev", "test")}
    man = manifest(cfg, splits, report, hashes, policy_hashes(), tokenizers_real=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    (out / "review_sample.md").write_text(review_sample(splits["test"], seed=cfg.seed))
    print(json.dumps({"dataset_sha256": man["dataset_sha256"], "dev": len(splits["dev"]),
                      "test": len(splits["test"]), "out": str(out)}, indent=2))
    return 0


def build(cfg: BuildConfig, s1, s2, counter: TokenCounter, smap) -> tuple[dict, list[str], dict]:
    b = Builder(cfg, s1, s2, counter, smap)
    splits, failures = b.build()
    report = checks.check_dataset(
        splits["dev"], splits["test"],
        min_pos_per_category=math.ceil(cfg.min_pos_category_share * cfg.test_size),
        min_pos_per_workload=math.ceil(cfg.min_pos_workload_share * cfg.test_size),
        min_hard_share=cfg.min_hard_share)
    for items in splits.values():
        for it in items:
            errs = checks.check_item(it, b.masked_originals.get(it["id"]))
            if errs:
                report["errors"].append(f"{it['id']}: {errs}")
    return splits, failures, report


if __name__ == "__main__":
    sys.exit(main())
