"""Setup: the Hugging Face token, the reasoning model, the live model catalog and the onboarding
of a Hugging Face model. Offline: moto for S3, DynamoDB and Secrets Manager; a fake Bedrock."""
from __future__ import annotations

import json

import boto3
import moto
import pytest

pytest.importorskip("fastapi")
from bench import catalog_store  # noqa: E402
from bench.config import ModelCatalog, load_models  # noqa: E402
from webui.api import onboarding as O  # noqa: E402
from webui.api import settings as S  # noqa: E402

SHA = "a" * 40
TOKEN = "hf_" + "Q" * 34


@pytest.fixture
def cloud(monkeypatch):
    """A bucket, the run table and the token secret, in moto; the app wired to them."""
    for k, v in (("AWS_DEFAULT_REGION", "us-east-1"), ("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing"),
                 ("AMXBENCH_CREDS_FILE", ""), ("BENCH_BUCKET", "amx-gpu-bench-123456789012-us-east-1")):
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AMXBENCH_OUTPUTS", raising=False)
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="amx-gpu-bench-123456789012-us-east-1")
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            TableName="amx-gpu-bench", BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}, {"AttributeName": "sk", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}, {"AttributeName": "sk", "KeyType": "RANGE"}])
        arn = boto3.client("secretsmanager", region_name="us-east-1").create_secret(Name="amx-gpu-bench/hf-token")["ARN"]
        monkeypatch.setenv("AMXBENCH_HF_SECRET_ARN", arn)
        S.reset_for_tests()
        catalog_store.invalidate()
        yield {"bucket": "amx-gpu-bench-123456789012-us-east-1", "secret_arn": arn}
    S.reset_for_tests()
    catalog_store.invalidate()


# ------------------------------------------------------------------------- Hugging Face token

def test_hf_token_is_validated_written_to_the_secret_and_never_read_back(cloud):
    with pytest.raises(ValueError, match="does not look like a Hugging Face token"):
        S.set_hf_token("ghp_notahuggingfacetoken", "me")
    out = S.set_hf_token(" " + TOKEN + "\n", "operator")
    assert out["configured"] and out["available"] and out["name"] == "amx-gpu-bench/hf-token" and out["last_changed"]
    assert TOKEN not in str(out)                                   # the status never carries the value
    assert boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId=cloud["secret_arn"])["SecretString"] == TOKEN
    item = S.get()
    assert item["updated_by"] == "operator" and item["hf_token_set_at"].endswith("Z") and "token" not in str(item).lower().replace("hf_token_set_at", "")
    assert S.hf_token_for_onboarding() == TOKEN                    # the onboarding job's download credential


def test_hf_token_status_without_a_wired_secret(cloud, monkeypatch):
    monkeypatch.delenv("AMXBENCH_HF_SECRET_ARN")
    st = S.hf_token_status()
    assert st == {"configured": False, "available": False, "reason": "no secret is wired to this deployment"}
    with pytest.raises(RuntimeError):
        S.set_hf_token(TOKEN, None)
    assert S.hf_token_for_onboarding() is None


# ------------------------------------------------------------------------- reasoning model

class FakeBedrock:
    MODELS = {
        "anthropic": [
            {"modelId": "anthropic.claude-opus-5-5", "modelName": "Claude Opus 5.5", "modelLifecycle": {"status": "ACTIVE"},
             "inferenceTypesSupported": ["INFERENCE_PROFILE"]},
            {"modelId": "anthropic.claude-opus-4-1-20250805-v1:0", "modelName": "Claude Opus 4.1", "modelLifecycle": {"status": "ACTIVE"},
             "inferenceTypesSupported": ["INFERENCE_PROFILE"]},
            {"modelId": "anthropic.claude-sonnet-4-5", "modelName": "Claude Sonnet 4.5", "modelLifecycle": {"status": "ACTIVE"},
             "inferenceTypesSupported": ["ON_DEMAND", "INFERENCE_PROFILE"]},
            {"modelId": "anthropic.claude-3-haiku", "modelName": "Claude 3 Haiku", "modelLifecycle": {"status": "LEGACY"},
             "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "anthropic.claude-opus-6-no-profile", "modelName": "Not callable", "modelLifecycle": {"status": "ACTIVE"},
             "inferenceTypesSupported": ["INFERENCE_PROFILE"]},
        ],
        "openai": [
            {"modelId": "openai.gpt-oss-120b-1:0", "modelName": "gpt-oss-120b", "modelLifecycle": {"status": "ACTIVE"},
             "inferenceTypesSupported": ["ON_DEMAND"]},
            {"modelId": "openai.text-embed-1", "modelName": "embed", "modelLifecycle": {"status": "ACTIVE"},
             "inferenceTypesSupported": ["ON_DEMAND"]},
        ],
        "amazon": [{"modelId": "amazon.nova-pro-v1:0", "modelName": "Nova Pro", "modelLifecycle": {"status": "ACTIVE"},
                    "inferenceTypesSupported": ["ON_DEMAND"]}],
    }

    def list_foundation_models(self, byProvider, byOutputModality):
        assert byOutputModality == "TEXT"
        return {"modelSummaries": self.MODELS.get(byProvider, [])}

    def list_inference_profiles(self, **kw):
        return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.anthropic.claude-opus-5-5"},
                                              {"inferenceProfileId": "us.anthropic.claude-opus-4-1-20250805-v1:0"},
                                              {"inferenceProfileId": "us.anthropic.claude-sonnet-4-5"}]}


def test_reasoning_models_are_claude_and_openai_only_newest_opus_first(cloud):
    models = S.list_reasoning_models(FakeBedrock())
    ids = [m["call_id"] for m in models]
    assert ids[0] == "us.anthropic.claude-opus-5-5"                              # newest Opus first
    assert "us.anthropic.claude-opus-4-1-20250805-v1:0" in ids and "anthropic.claude-sonnet-4-5" in ids
    assert "openai.gpt-oss-120b-1:0" in ids
    assert not any("nova" in i or "embed" in i or "haiku" in i or "no-profile" in i for i in ids), ids
    assert all(m["provider"] in S.ALLOWED_PROVIDERS for m in models)
    d = S.default_reasoning_model(models)
    assert d["call_id"] == "us.anthropic.claude-opus-5-5" and d["family"] == "opus"
    assert S.reasoning_model(FakeBedrock()) == {**d, "source": "default"}


def test_reasoning_model_choice_is_stored_validated_and_falls_back_when_gone(cloud):
    S.list_reasoning_models(FakeBedrock())
    with pytest.raises(ValueError, match="Pick a model from the list"):
        S.set_reasoning_model("amazon.nova-pro-v1:0", "me")
    out = S.set_reasoning_model("openai.gpt-oss-120b-1:0", "me")
    assert out["call_id"] == "openai.gpt-oss-120b-1:0" and out["source"] == "setup"
    assert S.get()["reasoning_model"]["name"] == "gpt-oss-120b" and S.get()["updated_by"] == "me"
    # The chosen model disappears from Bedrock: the default answers and says so.
    S.reset_for_tests()
    gone = FakeBedrock()
    gone.MODELS = {**FakeBedrock.MODELS, "openai": []}
    assert S.reasoning_model(gone)["source"] == "default"


# ------------------------------------------------------------------------- the live catalog

def test_catalog_store_seeds_from_the_repo_then_reads_what_setup_saved(cloud, monkeypatch):
    monkeypatch.setattr(catalog_store, "s3_factory", catalog_store._default_s3)
    seed = load_models(catalog_store.SEED)
    cat = catalog_store.load_catalog(fresh=True)
    assert [m.id for m in cat.models] == [m.id for m in seed.models] and catalog_store.source() == "seed"
    gone = [m.id for m in seed.models[2:]]
    smaller = ModelCatalog(size_cap_b=seed.size_cap_b, models=seed.models[:2], removed=gone)
    assert catalog_store.save_catalog(smaller) == "config/models.yaml"
    live = catalog_store.load_catalog(fresh=True)
    assert len(live.models) == 2 and catalog_store.source() == f"s3://{cloud['bucket']}/config/models.yaml"
    assert live.removed == sorted(gone)
    assert catalog_store.load_catalog() is live                    # cached within the TTL
    # A copy that merely LACKS shipped models (written before `removed` existed, or before a deploy that
    # shipped new models) gets them back from the seed: shipped definitions are code, not bucket state.
    catalog_store.save_catalog(ModelCatalog(size_cap_b=seed.size_cap_b, models=seed.models[:2]))
    assert [m.id for m in catalog_store.load_catalog(fresh=True).models] == [m.id for m in seed.models]
    # A broken object never takes the catalog down: the seed answers and the error is kept for the page.
    boto3.client("s3", region_name="us-east-1").put_object(Bucket=cloud["bucket"], Key="config/models.yaml", Body=b"models: [{id: x}]")
    assert len(catalog_store.load_catalog(fresh=True).models) == len(seed.models) and catalog_store.source() == "seed"
    assert "ValidationError" in catalog_store._cache["last_error"]


def test_default_bucket_comes_from_the_lambda_env_then_the_web_app_outputs(monkeypatch):
    for k in ("BENCH_BUCKET", "BUCKET", "AMXBENCH_OUTPUTS"):
        monkeypatch.delenv(k, raising=False)
    assert catalog_store.default_bucket() is None
    monkeypatch.setenv("AMXBENCH_OUTPUTS", '{"BucketName": "from-outputs"}')
    assert catalog_store.default_bucket() == "from-outputs"
    monkeypatch.setenv("BENCH_BUCKET", "from-lambda")
    assert catalog_store.default_bucket() == "from-lambda"


# ------------------------------------------------------------------------- onboarding: the spec

def found(repo="mistralai/Mistral-7B-Instruct-v0.3", params=7_248_023_552, files=(), **kw):
    return {"repo": repo, "revision": SHA, "gated": False, "license": "apache-2.0", "params": params,
            "config": {}, "files": set(files), **kw}


def test_repo_ids_and_model_ids():
    assert O.REPO_RE.match("mistralai/Mistral-7B-Instruct-v0.3") and O.REPO_RE.match("gpt2/x.y-z_1")
    for bad in ("mistral", "a/b/c", "/x", "x/", "a b/c", "-a/b"):
        assert not O.REPO_RE.match(bad), bad
    assert O.model_id_for("mistralai/Mistral-7B-Instruct-v0.3", set()) == "u-mistral-7b-instruct-v0.3"
    assert O.model_id_for("org/Name", {"u-name"}) == "u-name-2" and O.model_id_for("org/Name", {"u-name", "u-name-2"}) == "u-name-3"


def test_decoder_spec_gets_the_dspm_rules_of_the_shipped_chat_model_and_says_what_is_unproven():
    cfg = {"architectures": ["MistralForCausalLM"], "model_type": "mistral", "max_position_embeddings": 32768}
    spec, notes = O.derive_spec(found(), cfg, {"chat_template": "{{ x }}"}, {"m1"}, "U1")
    assert spec.id == "u-mistral-7b-instruct-v0.3" and spec.slot == "U1" and spec.type == "decoder"
    assert spec.user_added and spec.dspm_profile == "M1" and spec.suites == ["standard", "dspm", "translation", "intel"]
    assert spec.nominal_size_b == 7.25 and spec.max_input_tokens == 4096 and spec.max_model_len == 16384
    assert spec.prompt_format.kind == "chat_template"
    assert set(spec.precisions) == {"bf16", "bf16-ov", "int8-ov", "int4-ov"}
    assert spec.precisions["bf16"].engines == ["vllm", "thin-torch"] and spec.precisions["int4-ov"].engines == ["ovms"]
    assert notes[-1].startswith("Nothing has run this model here yet")
    assert not any("not on the support lists" in n for n in notes)       # mistral is on both lists
    assert not any("no chat template" in n for n in notes)
    # The catalog accepts it next to the shipped models once the cap is raised to fit.
    seed = load_models(catalog_store.SEED)
    cat = ModelCatalog(size_cap_b=max(seed.size_cap_b, spec.nominal_size_b), models=[*seed.models, spec])
    assert cat.get(spec.id).hf_repo == "mistralai/Mistral-7B-Instruct-v0.3"


def test_unknown_family_without_a_chat_template_is_added_with_plain_warnings():
    cfg = {"architectures": ["RWKVForCausalLM"], "model_type": "rwkv", "max_position_embeddings": 2048}
    spec, notes = O.derive_spec(found("org/rwkv-9b", params=9_000_000_000), cfg, {}, set(), "U2")
    assert spec.prompt_format.kind == "plain" and spec.max_input_tokens == 1536 and spec.max_model_len == 2048
    assert any("9 billion parameters" in n and "large for a CPU benchmark" in n for n in notes)
    assert any("no chat template" in n for n in notes)
    assert any("not on the support lists of ovms, vllm" in n for n in notes)


def test_encoder_decoder_and_classifier_specs():
    t5 = {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5", "is_encoder_decoder": True, "n_positions": 512}
    spec, _ = O.derive_spec(found("google/flan-t5-xl", params=2_850_000_000), t5, None, set(), "U1")
    assert spec.type == "encoder-decoder" and spec.dspm_profile == "M5" and set(spec.precisions) == {"bf16", "int8-ov"}
    assert spec.precisions["int8-ov"].engines == ["thin-openvino"] and spec.max_input_tokens == 512 and spec.max_model_len is None
    nli = {"architectures": ["DebertaV2ForSequenceClassification"], "model_type": "deberta-v2", "max_position_embeddings": 512,
           "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
    spec, notes = O.derive_spec(found("org/nli", params=184_000_000), nli, None, set(), "U1")
    assert spec.type == "encoder-classifier" and spec.dspm_profile is None and spec.suites == ["standard"]
    assert spec.prompt_format.kind == "pair" and spec.classifier.labels == ["entailment", "neutral", "contradiction"]
    assert spec.precisions["bf16"].inference_precision == "bf16" and "int8-ov" in spec.precisions
    assert any("speed benchmark only" in n for n in notes)
    tok = {"architectures": ["BertForTokenClassification"], "model_type": "bert", "id2label": {"0": "O", "1": "B-PER"}}
    spec, _ = O.derive_spec(found("org/ner", params=110_000_000), tok, None, set(), "U1")
    assert spec.type == "token-classifier" and spec.max_input_tokens == 512 and spec.precisions["bf16"].engines == ["thin-torch"]


def test_specs_the_benchmark_cannot_serve_are_refused_with_a_reason():
    with pytest.raises(O.OnboardingError, match="cannot serve this kind of model"):
        O.derive_spec(found(), {"architectures": ["CLIPModel"]}, None, set(), "U1")
    with pytest.raises(O.OnboardingError, match="does not report this model's size"):
        O.derive_spec(found(params=None), {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, None, set(), "U1")


def test_shipped_models_keep_their_own_dspm_rules_and_alternates_stand_in_for_their_slot():
    seed = load_models(catalog_store.SEED)
    for m in seed.models:
        assert not m.user_added
        base = m.slot.split("-")[0]
        if base in m.DSPM_SHIPPED:
            assert m.dspm_profile == base, (m.id, m.slot)   # M3-alt follows M3's rules
    # M6 (Gemma 4 E4B) shipped after the DSPM rules were written: a chat model, so it follows M1's rules.
    assert seed.get("m6-gemma-4-e4b").dspm_profile == "M1"
    assert seed.get("m7-gemma-4-e2b").dspm_profile == "M1"      # a chat model shipped after the rules: M1's


# ------------------------------------------------------------------------- onboarding: the job and remove

def test_one_job_at_a_time_and_removal_rules(cloud, monkeypatch):
    monkeypatch.setattr(catalog_store, "s3_factory", catalog_store._default_s3)
    O._jobs.clear()
    O._jobs["busy"] = {"id": "busy", "repo": "org/other", "status": "running", "stage": "x", "started_at": 0, "by": "a",
                       "notes": [], "error": None, "model": None}
    with pytest.raises(O.OnboardingError, match="org/other"):
        O.start("org/new", "me")
    s = O.status("busy")
    assert "by" not in s and s["elapsed_s"] > 0 and O.status("nope") is None
    O._jobs.clear()
    seed = load_models(catalog_store.SEED)
    with pytest.raises(O.OnboardingError, match="no such"):
        O.remove("u-missing", "me")
    out = O.remove(seed.models[0].id, "me")
    assert out == {"removed": seed.models[0].id, "remaining": len(seed.models) - 1}
    live = catalog_store.load_catalog(fresh=True)
    assert seed.models[0].id not in {m.id for m in live.models} and catalog_store.source().startswith("s3://")
    one = ModelCatalog(size_cap_b=seed.size_cap_b, models=[seed.models[1]],
                       removed=[m.id for m in seed.models if m.id != seed.models[1].id])
    catalog_store.save_catalog(one)
    with pytest.raises(O.OnboardingError, match="cannot be empty"):
        O.remove(seed.models[1].id, "me")


def test_live_catalog_takes_shipped_definitions_from_the_seed_and_keeps_setup_changes():
    seed = load_models(catalog_store.SEED)
    old_m1 = seed.models[0].model_copy(update={"suites": ["standard"], "vllm_args": []})   # an older deployment's copy
    added = seed.models[1].model_copy(update={"id": "u-added", "slot": "U1", "hf_repo": "org/added"})
    stored = ModelCatalog(size_cap_b=4.0, models=[old_m1, added], removed=["m7-gemma-4-e2b"])
    live = catalog_store.merge_with_seed(seed, stored)
    ids = [m.id for m in live.models]
    assert ids[0] == seed.models[0].id and live.models[0].suites == seed.models[0].suites      # the seed's definition
    assert "m7-gemma-4-e2b" not in ids                                                     # removed in Setup stays removed
    assert "m6-gemma-4-e4b" in ids                                                           # new in the seed: present
    assert ids[-1] == "u-added" and live.models[-1].hf_repo == "org/added"                   # Setup's addition kept
    assert live.size_cap_b == seed.size_cap_b == 8.0 and live.removed == ["m7-gemma-4-e2b"]


# ------------------------------------------------------------------------- the prompt corpus

def test_corpus_comes_from_the_bucket_and_must_match_the_pinned_text(tmp_path, monkeypatch):
    import hashlib

    from tools import build_prompts as bp

    real = {gid: (f"*** START OF x\nText {gid}. Another sentence here that is long enough. *** END OF x").encode() for gid in bp.GUTENBERG}
    monkeypatch.setattr(bp, "GUTENBERG_SHA256", {gid: hashlib.sha256(b).hexdigest() for gid, b in real.items()})
    monkeypatch.setattr(bp, "fetch", lambda url: (_ for _ in ()).throw(RuntimeError("HTTP Error 503")))   # gutenberg.org refuses

    class S3:
        def __init__(self, corrupt=()):
            self.calls, self.corrupt = [], set(corrupt)

        def get_object(self, Bucket, Key):
            self.calls.append(Key)
            gid = int(Key.rsplit("pg", 1)[1].split(".")[0])
            body = b"tampered" if gid in self.corrupt else real[gid]
            import io
            return {"Body": io.BytesIO(body)}

    s3 = S3()
    sentences, hashes = bp.gutenberg_corpus(tmp_path / "a", "bench-bucket", s3)
    assert sorted(s3.calls) == sorted(f"datasets/gutenberg/pg{g}.txt" for g in bp.GUTENBERG)
    assert hashes == {str(g): h for g, h in bp.GUTENBERG_SHA256.items()} and any("Another sentence" in x for x in sentences)
    # Cached: no second download.
    bp.gutenberg_corpus(tmp_path / "a", "bench-bucket", S3())
    # A text that does not match the pinned hash is refused (and not cached), with the reason.
    with pytest.raises(RuntimeError, match="does not match the pinned text"):
        bp.gutenberg_corpus(tmp_path / "b", "bench-bucket", S3(corrupt=[84]))
    assert not (tmp_path / "b" / "pg84.txt").exists()
    # No bucket and no origin: the error names both and the one-off publish command.
    with pytest.raises(RuntimeError, match="publish-corpus"):
        bp.gutenberg_corpus(tmp_path / "c", None, None)


def test_onboarding_merges_into_the_shared_requests_manifest(tmp_path):
    from tools.dspm.build_requests import write_manifest

    path = tmp_path / "manifest.json"
    shipped = {"pool_sizes": {"artifacts/dspm/v1/requests/d2-roblox-pii-v2/W3.jsonl": 1600},
               "pools": {"artifacts/dspm/v1/requests/d2-roblox-pii-v2/W3.jsonl": {"n": 1600}}, "policy": {"version": 1}, "seed": 1}
    write_manifest(path, shipped, merge=False)
    added = {"pool_sizes": {"artifacts/dspm/v1/requests/u-x/W3.jsonl": 1600}, "pools": {"artifacts/dspm/v1/requests/u-x/W3.jsonl": {"n": 1600}},
             "policy": {"version": 1}, "seed": 1, "dataset_manifest_sha256": "abc"}
    out = write_manifest(path, added, merge=True)                      # the Setup page's add
    assert set(out["pool_sizes"]) == {"artifacts/dspm/v1/requests/d2-roblox-pii-v2/W3.jsonl", "artifacts/dspm/v1/requests/u-x/W3.jsonl"}
    assert set(json.loads(path.read_text())["pools"]) == set(out["pool_sizes"]) and out["dataset_manifest_sha256"] == "abc"
    assert set(write_manifest(path, added, merge=False)["pool_sizes"]) == {"artifacts/dspm/v1/requests/u-x/W3.jsonl"}   # the CLI rebuild
