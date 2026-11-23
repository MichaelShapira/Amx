"""CoreStack (design 3, 6, 9, 10): bucket, table, images, Lambdas with least-privilege
roles, the run and level state machines, the reaper schedule and the host scripts."""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, SecretValue, Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_deployment as s3deploy
from aws_cdk import aws_secretsmanager as sm
from aws_cdk import aws_ssm as ssm
from aws_cdk import aws_stepfunctions as sfn
from aws_cdk import aws_stepfunctions_tasks as tasks
from aws_cdk import custom_resources as cr
from constructs import Construct

from bench.deployment import HF_SECRET_NAME, HF_TOKEN_UNSET
from infra.stacks.bundling import lambda_code
from infra.stacks.network import NetworkStack

ROOT = Path(__file__).resolve().parents[2]
WORKER_RESERVED = 272         # Intel's benchmark sweeps to 256 concurrent requests (one worker each) plus headroom
def _built_images() -> dict[str, tuple[str, dict]]:
    """name -> (Dockerfile, build args) for every engine image this project builds, from
    config/images.yaml (the web app's Technical details page shows the same file)."""
    import yaml

    out = {}
    for name, spec in yaml.safe_load((ROOT / "config" / "images.yaml").read_text())["images"].items():
        if spec.get("runs_on", "").startswith("ECS"):
            continue                              # the web app image is the WebApp stack's asset
        base = spec.get("base_image", "")
        args = {"BASE_IMAGE": base} if "@sha256:" in base else {}
        out[name] = (spec["dockerfile"], args)
    return out


IMAGES = _built_images()
P = sfn.JsonPath


class CoreStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, network: NetworkStack, skip_bundling: bool = False,
                 build_images: bool = True, harness_commit: str = "unknown", **kw):
        super().__init__(scope, cid, **kw)
        self.network = network
        self.bucket = s3.Bucket(
            self, "Bucket", bucket_name=f"amx-gpu-bench-{self.account}-{self.region}",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL, enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED, object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            removal_policy=RemovalPolicy.RETAIN, minimum_tls_version=1.2)
        self.table = ddb.Table(
            self, "Table", table_name="amx-gpu-bench", billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            partition_key=ddb.Attribute(name="pk", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True),
            removal_policy=RemovalPolicy.RETAIN)
        # The Hugging Face token slot. Created here with a placeholder; Setup -> Hugging Face token writes the real value
        # (the web app reads the value to tell "unset" from a token). A deployment whose secret predates this stack
        # imports it instead: -c hf_secret_arn=arn:aws:secretsmanager:...:secret:amx-gpu-bench/hf-token-XXXXXX.
        existing = self.node.try_get_context("hf_secret_arn")
        if existing:
            self.hf_secret = sm.Secret.from_secret_complete_arn(self, "HfToken", str(existing))
        else:
            self.hf_secret = sm.Secret(self, "HfToken", secret_name=HF_SECRET_NAME, removal_policy=RemovalPolicy.RETAIN,
                                       description="Hugging Face access token for gated models (set from the web app's Setup page)",
                                       secret_string_value=SecretValue.unsafe_plain_text(HF_TOKEN_UNSET))

        # Thin-server and export images (built at deploy time, pushed to the CDK asset repository).
        self.image_assets: dict[str, ecr_assets.DockerImageAsset] = {}
        tags: dict[str, str] = {}
        if build_images:
            for name, (dockerfile, args) in IMAGES.items():
                a = ecr_assets.DockerImageAsset(self, f"Image{name.title().replace('-', '')}", directory=str(ROOT),
                                                file=dockerfile, build_args=args or None,
                                                platform=ecr_assets.Platform.LINUX_AMD64)
                self.image_assets[name] = a
                tags[name] = a.image_tag
        first = next(iter(self.image_assets.values()), None)
        self.ecr_repo_uri = first.repository.repository_uri if first else ""
        ssm.StringParameter(self, "ImagesParam", parameter_name="/amx-gpu-bench/images",
                            string_value=self.to_json_string(tags) if tags else "{}")

        # Host scripts -> s3://bucket/artifacts/host/ (instances sync them at boot).
        s3deploy.BucketDeployment(self, "HostScripts", destination_bucket=self.bucket,
                                  destination_key_prefix="artifacts/host/",
                                  sources=[s3deploy.Source.asset(str(ROOT / "host"))], prune=True,
                                  memory_limit=256)

        code = lambda_code(skip_bundling)
        self.fns: dict[str, lambda_.Function] = {}
        common_env = {"BENCH_BUCKET": self.bucket.bucket_name, "BENCH_TABLE": self.table.table_name,
                      "BENCH_HOME": "/opt/bench", "SSM_DOCUMENT": "AWS-RunShellScript"}

        def fn(name: str, handler: str, *, vpc: bool = False, memory: int = 512, timeout_min: int = 15,
               reserved: int | None = None, env: dict | None = None) -> lambda_.Function:
            fname = f"amx-gpu-bench-{name.replace('_', '-')}"
            lg = logs.LogGroup(self, f"Logs{name.title().replace('_', '')}", log_group_name=f"/aws/lambda/{fname}",
                               retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.DESTROY)
            f = lambda_.Function(
                self, f"Fn{name.title().replace('_', '')}", function_name=fname,
                runtime=lambda_.Runtime.PYTHON_3_12, architecture=lambda_.Architecture.ARM_64, code=code,
                handler=handler, memory_size=memory, timeout=Duration.minutes(timeout_min),
                environment={**common_env, **(env or {})}, reserved_concurrent_executions=reserved,
                log_group=lg,
                vpc=network.vpc if vpc else None, vpc_subnets=network.private_subnets if vpc else None,
                security_groups=[network.sg_workers] if vpc else None)
            self.fns[name] = f
            return f

        worker = fn("worker", "lambdas.worker.handler", vpc=True, memory=1769, reserved=WORKER_RESERVED)
        warmup = fn("warmup", "lambdas.side_tasks.warmup", vpc=True, memory=1769)
        quality = fn("quality", "lambdas.side_tasks.quality", vpc=True, memory=1769)
        init_level = fn("init_level", "lambdas.waves.init_level", memory=256, timeout_min=1)
        plan_wave = fn("plan_wave", "lambdas.waves.plan_wave", memory=512, timeout_min=1)
        wave_check = fn("wave_check", "lambdas.waves.wave_check", memory=256, timeout_min=1)
        perf = fn("perf_counters", "lambdas.side_tasks.perf_counters", memory=256, timeout_min=5)
        amx_probe = fn("amx_probe", "lambdas.side_tasks.amx_probe", memory=512)
        membw = fn("membw_probe", "lambdas.side_tasks.membw_probe", memory=256)
        agg_level = fn("aggregate_level", "lambdas.aggregate.aggregate_level", memory=1769)
        agg_run = fn("aggregate_run", "lambdas.aggregate.aggregate_run", memory=3008)
        self.private_subnet_id = network.vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS).subnet_ids[0]
        control = fn("target_control", "lambdas.target_control.handler", memory=256)
        reaper = fn("reaper", "lambdas.reaper.handler", memory=256, timeout_min=5,
                    env={"IDLE_MINUTES": "30", "MAX_UPTIME_HOURS": "8"})
        preflight = fn("preflight", "lambdas.preflight.handler", memory=1024, timeout_min=5,
                       env={"WORKER_FUNCTION_NAME": "amx-gpu-bench-worker", "HF_TOKEN_SECRET_ARN": self.hf_secret.secret_arn,
                            "ECR_REPOSITORY_URI": self.ecr_repo_uri, "HARNESS_COMMIT": harness_commit})

        self._grant(worker, warmup, quality, init_level, plan_wave, wave_check, perf, amx_probe, membw, agg_level,
                    agg_run, control, reaper, preflight)
        self.level_sm = self._level_state_machine(init_level, warmup, perf, plan_wave, worker, wave_check, agg_level)
        self.run_sm = self._run_state_machine(preflight, control, amx_probe, membw, quality, agg_run)
        self._activate_cost_allocation_tag(str(self.node.try_get_context("cost_tag_revision") or "1"))
        self.preflight_fn = preflight                     # the web app reads its HARNESS_COMMIT
        events.Rule(self, "ReaperSchedule", schedule=events.Schedule.rate(Duration.minutes(10)),
                    targets=[targets.LambdaFunction(reaper)])
        for k, v in {"BucketName": self.bucket.bucket_name, "TableName": self.table.table_name,
                     "RunStateMachineArn": self.run_sm.state_machine_arn,
                     "LevelStateMachineArn": self.level_sm.state_machine_arn,
                     "EcrRepositoryUri": self.ecr_repo_uri or "none"}.items():
            CfnOutput(self, k, value=v)

    # ------------------------------------------------------------------ IAM
    def _obj(self, *patterns: str) -> list[str]:
        return [self.bucket.arn_for_objects(p) for p in patterns]

    def _activate_cost_allocation_tag(self, revision: str) -> None:
        """Turn the project's tag key into an AWS cost allocation tag (Cost Explorer, CUR), so the whole project's
        spend groups under it. Billing only knows a key once a resource carrying it has been billed (up to a day
        after the first deploy), and the API refuses an unknown key with a ValidationException, so that error is
        tolerated here and the call is repeated on demand: `cdk deploy AmxBench-Core -c cost_tag_revision=<N+1>`,
        or by hand `aws ce update-cost-allocation-tags-status --cost-allocation-tags-status TagKey=Amx,Status=Active`.
        Cost Explorer actions take no resource ARN (the policy below is the narrowest grant there is)."""
        from bench.tags import COST_ALLOCATION_TAG
        call = cr.AwsSdkCall(
            service="CostExplorer", action="updateCostAllocationTagsStatus",
            parameters={"CostAllocationTagsStatus": [{"TagKey": COST_ALLOCATION_TAG, "Status": "Active"}],
                        "Revision": revision},    # a changed property re-runs the call on the next deploy
            physical_resource_id=cr.PhysicalResourceId.of(f"cost-allocation-tag-{COST_ALLOCATION_TAG}"),
            region="us-east-1",                   # Cost Explorer's single endpoint
            ignore_error_codes_matching="ValidationException")
        cr.AwsCustomResource(
            self, "CostAllocationTag", on_create=call, on_update=call, install_latest_aws_sdk=False,
            policy=cr.AwsCustomResourcePolicy.from_statements([iam.PolicyStatement(
                actions=["ce:UpdateCostAllocationTagsStatus"], resources=["*"])]),
            log_group=logs.LogGroup(self, "LogsCostAllocationTag", log_group_name="/aws/lambda/amx-gpu-bench-cost-allocation-tag",
                                    retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.DESTROY),
            timeout=Duration.minutes(2))

    def _grant(self, worker, warmup, quality, init_level, plan_wave, wave_check, perf, amx_probe, membw, agg_level,
               agg_run, control, reaper, preflight) -> None:
        t = self.table
        s3_get = lambda *p: iam.PolicyStatement(actions=["s3:GetObject"], resources=self._obj(*p))  # noqa: E731
        s3_put = lambda *p: iam.PolicyStatement(actions=["s3:PutObject"], resources=self._obj(*p))  # noqa: E731
        s3_list = iam.PolicyStatement(actions=["s3:ListBucket"], resources=[self.bucket.bucket_arn])
        for f in (worker, warmup, quality):
            f.add_to_role_policy(s3_get("artifacts/*"))
        worker.add_to_role_policy(s3_put("runs/*/raw/*"))
        warmup.add_to_role_policy(s3_put("runs/*/warmup/*"))
        quality.add_to_role_policy(s3_put("runs/*/quality/*"))
        worker.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        for f in (init_level, plan_wave, wave_check):
            f.add_to_role_policy(iam.PolicyStatement(
                actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"], resources=[t.table_arn]))
        # A re-run of a level (after a Spot interruption) first drops the raw records of the cut-short attempt.
        init_level.add_to_role_policy(s3_list)
        init_level.add_to_role_policy(iam.PolicyStatement(actions=["s3:DeleteObject"], resources=self._obj("runs/*/raw/*")))
        ssm_send = iam.PolicyStatement(
            actions=["ssm:SendCommand"],
            resources=[f"arn:aws:ssm:{self.region}::document/AWS-RunShellScript"])
        ssm_send_inst = iam.PolicyStatement(
            actions=["ssm:SendCommand"], resources=[f"arn:aws:ec2:{self.region}:{self.account}:instance/*"],
            conditions={"StringEquals": {"ssm:resourceTag/bench:managed": "true"}})
        # Read-only follow-up of the commands sent above. Neither action supports a resource ARN or a tag condition
        # (IAM reference: GetCommandInvocation and Describe* take only "*"); the write side, SendCommand, is pinned to
        # the one document and to bench-managed instances.
        ssm_read = iam.PolicyStatement(actions=["ssm:GetCommandInvocation", "ssm:DescribeInstanceInformation"],
                                       resources=["*"])
        for f in (perf, amx_probe, membw, control):
            for st in (ssm_send, ssm_send_inst, ssm_read):
                f.add_to_role_policy(st)
        amx_probe.add_to_role_policy(s3_get("runs/*/amx/*"))
        amx_probe.add_to_role_policy(s3_put("runs/*/amx/*"))
        ec2_tagged = {"StringEquals": {"aws:ResourceTag/bench:managed": "true"}}
        inst_arn = f"arn:aws:ec2:{self.region}:{self.account}:instance/*"
        for f in (control, reaper):
            # ec2:Describe* supports no resource ARN and no aws:ResourceTag condition (a condition would deny every
            # call); the actions that change state below are all conditioned on the bench:managed tag.
            f.add_to_role_policy(iam.PolicyStatement(
                actions=["ec2:DescribeInstances", "ec2:DescribeLaunchTemplateVersions"], resources=["*"]))
            # Stop On-Demand instances, terminate one-time Spot ones; only bench-tagged instances either way.
            f.add_to_role_policy(iam.PolicyStatement(actions=["ec2:StopInstances", "ec2:TerminateInstances"],
                                                     resources=[inst_arn], conditions=ec2_tagged))
            f.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:GetItem", "dynamodb:UpdateItem"],
                                                     resources=[t.table_arn]))
        control.add_to_role_policy(iam.PolicyStatement(actions=["ec2:StartInstances"], resources=[inst_arn],
                                                       conditions=ec2_tagged))
        # Launch per run from a target stack's launch template (On-Demand or Spot), in the private subnet
        # with the targets security group; tags are set in the same call.
        control.add_to_role_policy(iam.PolicyStatement(
            actions=["ec2:RunInstances"],
            resources=[inst_arn, f"arn:aws:ec2:{self.region}:{self.account}:volume/*",
                       f"arn:aws:ec2:{self.region}:{self.account}:network-interface/*",
                       f"arn:aws:ec2:{self.region}:{self.account}:launch-template/*",
                       f"arn:aws:ec2:{self.region}::image/*",
                       f"arn:aws:ec2:{self.region}:{self.account}:subnet/{self.private_subnet_id}",
                       f"arn:aws:ec2:{self.region}:{self.account}:security-group/"
                       f"{self.network.sg_targets.security_group_id}"]))
        control.add_to_role_policy(iam.PolicyStatement(
            actions=["ec2:CreateTags"], resources=[inst_arn, f"arn:aws:ec2:{self.region}:{self.account}:volume/*"],
            conditions={"StringEquals": {"ec2:CreateAction": "RunInstances"}}))
        control.add_to_role_policy(iam.PolicyStatement(
            actions=["iam:PassRole"], resources=[f"arn:aws:iam::{self.account}:role/amx-gpu-bench-target-*"],
            conditions={"StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}}))
        control.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/amx-gpu-bench/targets/*"]))
        control.add_to_role_policy(s3_get("runs/*"))
        control.add_to_role_policy(s3_put("runs/*/capacity/*", "runs/*/targets/*", "runs/*/plan/*"))
        agg_level.add_to_role_policy(s3_list)
        agg_level.add_to_role_policy(s3_get("runs/*"))
        agg_level.add_to_role_policy(s3_put("runs/*/metrics/*", "runs/*/predictions/*"))
        # Host utilization per measurement (AmxGpuBench namespace). GetMetricData supports no resource ARN and no
        # condition key (cloudwatch:namespace applies to PutMetricData only): read-only, "*".
        agg_level.add_to_role_policy(iam.PolicyStatement(actions=["cloudwatch:GetMetricData"], resources=["*"]))
        agg_level.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        agg_run.add_to_role_policy(s3_list)
        # datasets/*: the DSPM labels; artifacts/prompts/*: the decision pools carry their gold labels (plan.decision_pool).
        agg_run.add_to_role_policy(s3_get("runs/*", "datasets/*", "artifacts/prompts/*", "config/models.yaml"))
        agg_run.add_to_role_policy(s3_put("runs/*/report/*"))
        agg_run.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        preflight.add_to_role_policy(s3_list)
        preflight.add_to_role_policy(s3_get("runs/*", "hosts/*", "artifacts/*", "datasets/dspm/v1/manifest.json",
                                            "datasets/translation/v1/manifest.json",
                                            "config/models.yaml"))
        preflight.add_to_role_policy(s3_put("runs/*"))
        preflight.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        preflight.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/amx-gpu-bench/*"]))
        preflight.add_to_role_policy(iam.PolicyStatement(
            actions=["lambda:GetFunctionConcurrency"], resources=[self.fns["worker"].function_arn]))
        # DescribeInstanceTypeOfferings takes only "*" (no resource type). The price list is read for
        # EC2 only, and the pricing actions accept no condition keys either, so the scope is the service.
        preflight.add_to_role_policy(iam.PolicyStatement(
            actions=["ec2:DescribeInstanceTypeOfferings", "pricing:GetProducts", "pricing:ListPriceLists"],
            resources=["*"]))
        self.hf_secret.grant_read(preflight)

    # ------------------------------------------------------------------ level state machine
    def _level_state_machine(self, init_level, warmup, perf, plan_wave, worker, wave_check, agg_level):
        def invoke(cid, f, payload, **kw):
            return tasks.LambdaInvoke(self, cid, lambda_function=f, payload=sfn.TaskInput.from_object(payload),
                                      retry_on_service_exceptions=True, **kw)

        start = sfn.Pass(self, "LevelStart", parameters={"level.$": "$.level", "wave": 0, "retries": 0})
        init = invoke("InitLevel", init_level, {"level.$": "$.level"}, result_path=P.DISCARD)
        warm = invoke("Warmup", warmup, {"level.$": "$.level"},
                      result_selector={"max_items_per_worker.$": "$.Payload.max_items_per_worker"},
                      result_path="$.warm")
        merge = sfn.Pass(self, "MergeWarmup", parameters={
            "level.$": "States.JsonMerge($.level, $.warm, false)", "wave.$": "$.wave", "retries.$": "$.retries"})
        perf_start = invoke("PerfStart", perf, {"action": "start", "level.$": "$.level"}, result_path=P.DISCARD)
        plan = invoke("PlanWave", plan_wave, {"level.$": "$.level", "wave.$": "$.wave"},
                      result_selector={"items.$": "$.Payload.items", "wave.$": "$.Payload.wave"}, result_path="$.plan")

        def wave_map(cid: str, warm_mode: bool) -> sfn.DistributedMap:
            m = sfn.DistributedMap(
                self, cid, items_path="$.plan.items", max_concurrency_path="$.level.conc",
                map_execution_type=sfn.StateMachineType.STANDARD, tolerated_failure_percentage=0,
                item_selector={"mode": "warm"} if warm_mode else None, result_path=P.DISCARD)
            m.item_processor(tasks.LambdaInvoke(self, f"{cid}Worker", lambda_function=worker,
                                                payload=sfn.TaskInput.from_json_path_at("$"),
                                                retry_on_service_exceptions=True, result_selector={"ok": True}))
            return m

        prewarm = wave_map("PrewarmWave", True)
        measure = wave_map("MeasureWave", False)
        check = invoke("WaveCheck", wave_check, {"level.$": "$.level", "wave.$": "$.wave"},
                       result_selector={"more.$": "$.Payload.more", "wave.$": "$.Payload.wave"}, result_path="$.check")
        next_wave = sfn.Pass(self, "NextWave", parameters={"level.$": "$.level", "wave.$": "$.check.wave",
                                                           "retries.$": "$.retries"})
        retry_wave = sfn.Pass(self, "RetryWave", parameters={"level.$": "$.level", "wave.$": "$.wave", "retries": 1})
        # Every failure path records the error on the level item first (the run state machine
        # reads it to decide whether a relaunch must re-run the level), then fails the execution.
        mark_failed = invoke("MarkLevelFailed", init_level, {"action": "mark_failed", "level.$": "$.level",
                                                             "error.$": "$.error"}, result_path=P.DISCARD)
        fail = sfn.Fail(self, "LevelFailed", cause="see the level item's error", error="LevelFailed")
        mark_failed.next(fail)
        mark_failed.add_catch(fail, result_path=P.DISCARD)
        wave_failed = sfn.Pass(self, "WaveFailedTwice", result=sfn.Result.from_object({"Error": "WaveFailed"}),
                               result_path="$.error")
        wave_failed.next(mark_failed)
        perf_stop = invoke("PerfStop", perf, {"action": "stop", "level.$": "$.level"},
                           result_selector={"counts.$": "$.Payload.counts"}, result_path="$.perf")
        no_perf = sfn.Pass(self, "NoPerf", result=sfn.Result.from_object({"counts": None}), result_path="$.perf")
        aggregate = invoke("AggregateLevel", agg_level, {"level.$": "$.level", "perf.$": "$.perf"},
                           result_selector={"n_ok.$": "$.Payload.n_ok"}, result_path="$.result")
        perf_start.add_catch(plan, result_path=P.DISCARD)
        perf_stop.add_catch(no_perf, result_path=P.DISCARD)
        measure.add_catch(sfn.Choice(self, "WaveFailedOnce").when(
            sfn.Condition.number_equals("$.retries", 0), retry_wave.next(plan)).otherwise(wave_failed),
            result_path=P.DISCARD)
        for st in (init, warm, plan, prewarm, check, aggregate):
            st.add_catch(mark_failed, result_path="$.error")
        more = sfn.Choice(self, "MoreWaves").when(sfn.Condition.boolean_equals("$.check.more", True), next_wave.next(plan))
        is_cpu_end = sfn.Choice(self, "CpuLevelEnd").when(sfn.Condition.boolean_equals("$.level.cpu", True), perf_stop)
        more.otherwise(is_cpu_end)
        is_cpu_end.otherwise(no_perf)
        perf_stop.next(aggregate)
        no_perf.next(aggregate)
        first = sfn.Choice(self, "FirstWave").when(sfn.Condition.number_equals("$.wave", 0), prewarm.next(measure)) \
            .otherwise(measure)
        measure.next(check)
        check.next(more)
        plan.next(first)
        is_cpu_start = sfn.Choice(self, "CpuLevelStart").when(sfn.Condition.boolean_equals("$.level.cpu", True),
                                                              perf_start.next(plan)).otherwise(plan)
        definition = start.next(init).next(warm).next(merge).next(is_cpu_start)
        return sfn.StateMachine(self, "LevelStateMachine", state_machine_name="amx-gpu-bench-level",
                                definition_body=sfn.DefinitionBody.from_chainable(definition),
                                state_machine_type=sfn.StateMachineType.STANDARD, timeout=Duration.hours(12),
                                tracing_enabled=False)

    # ------------------------------------------------------------------ run state machine
    def _run_state_machine(self, preflight, control, amx_probe, membw, quality, agg_run):
        def invoke(cid, f, payload, **kw):
            return tasks.LambdaInvoke(self, cid, lambda_function=f, payload=sfn.TaskInput.from_object(payload),
                                      retry_on_service_exceptions=True, **kw)

        pf = invoke("Preflight", preflight, {"run_id.$": "$.run_id"},
                    result_selector={"run_id.$": "$.Payload.run_id", "targets.$": "$.Payload.targets",
                                     "purchase.$": "$.Payload.purchase",
                                     "parallel.$": "$.Payload.max_target_concurrency",
                                     "stop.$": "$.Payload.stop_targets_after_run"}, result_path="$.pf")
        # Per cell. The cell item carries run_id, target_id, purchase, cell, instance_id, private_ip and
        # recovery {attempt, interruption_logged}; a failed step goes to Check, which relaunches a gone
        # (Spot-interrupted) instance and resumes the cell's remaining levels on the replacement.
        cstart = invoke("ConfigureStart", control, {"action": "configure_start", "run_id.$": "$.run_id",
                                                     "instance_id.$": "$.instance_id", "cell.$": "$.cell"},
                        result_selector={"command_id.$": "$.Payload.command_id"}, result_path="$.cmd")
        cpoll = invoke("ConfigurePoll", control, {
            "action": "configure_poll", "run_id.$": "$.run_id", "instance_id.$": "$.instance_id",
            "target_id.$": "$.target_id", "private_ip.$": "$.private_ip", "cell.$": "$.cell",
            "command_id.$": "$.cmd.command_id"}, result_selector={"r.$": "$.Payload"}, result_path="$.cfg")
        amx = invoke("AmxProbe", amx_probe, {"run_id.$": "$.run_id", "instance_id.$": "$.instance_id",
                                             "cell.$": "$.cell"}, result_path=P.DISCARD)
        mem = invoke("MemBwProbe", membw, {"run_id.$": "$.run_id", "instance_id.$": "$.instance_id"},
                     result_path=P.DISCARD)
        qual = invoke("Quality", quality, {"quality.$": "$.cfg.r.quality"}, result_path=P.DISCARD)
        # One failed level fails the map so that Check can tell an interrupted instance from a failed level.
        levels = sfn.DistributedMap(
            self, "Levels", item_reader=sfn.S3JsonItemReader(bucket=self.bucket, key=P.string_at("$.cfg.r.levels_key")),
            max_concurrency=1, map_execution_type=sfn.StateMachineType.STANDARD, tolerated_failure_percentage=0,
            item_selector={"level.$": "$$.Map.Item.Value"}, result_path=P.DISCARD)
        levels.item_processor(tasks.StepFunctionsStartExecution(
            self, "RunLevel", state_machine=self.level_sm, integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            input=sfn.TaskInput.from_json_path_at("$"), result_selector={"ok": True}))
        cell_done = sfn.Pass(self, "CellDone")
        cell_failed = sfn.Pass(self, "CellFailed", parameters={"failed.$": "$.cell.cell_id", "error.$": "$.error"})
        after_mem = sfn.Choice(self, "NeedsQuality").when(sfn.Condition.is_not_null("$.cfg.r.quality"),
                                                          qual.next(levels)).otherwise(levels)
        after_amx = sfn.Choice(self, "NeedsMemBw").when(sfn.Condition.boolean_equals("$.cell.membw", True),
                                                        mem.next(after_mem)).otherwise(after_mem)
        after_cfg = sfn.Choice(self, "NeedsAmx").when(sfn.Condition.boolean_equals("$.cell.amx", True),
                                                      amx.next(after_amx)).otherwise(after_amx)
        ready = sfn.Choice(self, "CellReady").when(sfn.Condition.boolean_equals("$.cfg.r.done", True), after_cfg) \
            .otherwise(cpoll)
        levels.next(cell_done)
        # Recovery: which stage failed, then Check decides relaunch / resume / fail.
        stage_bringup = sfn.Pass(self, "FailedInBringup", result=sfn.Result.from_string("bringup"), result_path="$.stage")
        stage_levels = sfn.Pass(self, "FailedInLevels", result=sfn.Result.from_string("levels"), result_path="$.stage")
        check = invoke("Check", control, {
            "action": "check", "run_id.$": "$.run_id", "target_id.$": "$.target_id", "instance_id.$": "$.instance_id",
            "private_ip.$": "$.private_ip", "cell.$": "$.cell", "purchase.$": "$.purchase",
            "alternates.$": "$.alternates", "instance_type.$": "$.instance_type_requested",
            "recovery.$": "$.recovery", "stage.$": "$.stage", "error.$": "$.error"},
            result_selector={"r.$": "$.Payload"}, result_path="$.chk")
        adopt = sfn.Pass(self, "AdoptCheck", parameters={
            "run_id.$": "$.run_id", "target_id.$": "$.target_id", "purchase.$": "$.purchase", "cell.$": "$.cell",
            "alternates.$": "$.alternates", "instance_type_requested.$": "$.instance_type_requested",
            "instance_id.$": "$.chk.r.instance_id", "private_ip.$": "$.chk.r.private_ip",
            "recovery.$": "$.chk.r.recovery", "decision.$": "$.chk.r.decision", "ready.$": "$.chk.r.ready",
            "stage.$": "$.stage", "error.$": "$.error"})
        wait_replace = sfn.Wait(self, "WaitForReplacement", time=sfn.WaitTime.duration(Duration.seconds(60)))
        resume = invoke("ResumeLevels", control, {
            "action": "resume_levels", "run_id.$": "$.run_id", "target_id.$": "$.target_id",
            "instance_id.$": "$.instance_id", "private_ip.$": "$.private_ip", "cell.$": "$.cell"},
            result_selector={"r.$": "$.Payload"}, result_path="$.cfg")
        decide = sfn.Choice(self, "AfterCheck") \
            .when(sfn.Condition.string_equals("$.decision", "fail"), cell_failed) \
            .when(sfn.Condition.boolean_equals("$.ready", False), wait_replace.next(check)) \
            .when(sfn.Condition.string_equals("$.decision", "relaunch"), cstart) \
            .otherwise(resume.next(levels))
        check.next(adopt).next(decide)
        check.add_catch(cell_failed, result_path="$.error")
        resume.add_catch(cell_failed, result_path="$.error")
        for st in (cstart, cpoll, amx, mem, qual):
            st.add_catch(stage_bringup, result_path="$.error")
        levels.add_catch(stage_levels, result_path="$.error")
        stage_bringup.next(check)
        stage_levels.next(check)
        cells = sfn.Map(self, "Cells", items_path="$.target.cells", max_concurrency=1, result_path=P.DISCARD,
                        item_selector={"run_id.$": "$.run_id", "target_id.$": "$.target.target_id",
                                       "purchase.$": "$.purchase", "alternates.$": "$.target.alternates",
                                       "instance_type_requested.$": "$.target.instance_type",
                                       "instance_id.$": "$.inst.r.instance_id",
                                       "private_ip.$": "$.inst.r.private_ip", "cell.$": "$$.Map.Item.Value",
                                       "recovery": {"attempt": 0, "interruption_logged": False}})
        cells.item_processor(cstart.next(cpoll).next(ready))
        # Launch or start the target's instance; loop while capacity is short or the instance boots.
        # `cells` lets a give-up (purchase.max_launch_attempts refused launches) mark every cell of the
        # target as not_run with the reason, instead of leaving them 'planned' with no explanation.
        ensure = invoke("EnsureRunning", control, {"action": "ensure_running", "run_id.$": "$.run_id",
                                                    "target_id.$": "$.target.target_id", "purchase.$": "$.purchase",
                                                    "alternates.$": "$.target.alternates",
                                                    "instance_type.$": "$.target.instance_type",
                                                    "cells.$": "$.target.cells"},
                        result_selector={"r.$": "$.Payload"}, result_path="$.inst")
        wait_launch = sfn.Wait(self, "WaitForInstance", time=sfn.WaitTime.duration(Duration.seconds(60)))
        target_failed = sfn.Pass(self, "TargetFailed")
        ensure.add_catch(target_failed, result_path="$.error")
        running = sfn.Choice(self, "InstanceReady").when(sfn.Condition.boolean_equals("$.inst.r.ready", True), cells) \
            .otherwise(wait_launch.next(ensure))
        tmap = sfn.Map(self, "Targets", items_path="$.pf.targets", max_concurrency_path="$.pf.parallel",
                       result_path=P.DISCARD,
                       item_selector={"run_id.$": "$.pf.run_id", "purchase.$": "$.pf.purchase",
                                      "target.$": "$$.Map.Item.Value"})
        tmap.item_processor(ensure.next(running))
        agg = invoke("AggregateRun", agg_run, {"run_id.$": "$.pf.run_id"},
                     result_selector={"report_prefix.$": "$.Payload.report_prefix"}, result_path="$.report")
        stop_one = invoke("StopTarget", control, {"action": "stop", "target_id.$": "$.target_id", "run_id.$": "$.run_id"},
                          result_path=P.DISCARD)
        stop_map = sfn.Map(self, "StopTargets", items_path="$.pf.targets", result_path=P.DISCARD,
                           item_selector={"target_id.$": "$$.Map.Item.Value.target_id", "run_id.$": "$.pf.run_id"})
        stop_map.item_processor(stop_one)
        done = sfn.Succeed(self, "RunDone")
        stop_choice = sfn.Choice(self, "StopAfterRun").when(sfn.Condition.boolean_equals("$.pf.stop", True),
                                                            stop_map.next(done)).otherwise(done)
        agg.add_catch(stop_choice, result_path="$.aggregate_error")
        tmap.add_catch(agg, result_path="$.targets_error")
        definition = pf.next(tmap).next(agg).next(stop_choice)
        sm_ = sfn.StateMachine(self, "RunStateMachine", state_machine_name="amx-gpu-bench-run",
                               definition_body=sfn.DefinitionBody.from_chainable(definition),
                               state_machine_type=sfn.StateMachineType.STANDARD, timeout=Duration.days(3))
        self.bucket.grant_read(sm_, "runs/*")
        return sm_

    def instance_policy(self, role: iam.IRole, target_id: str) -> None:
        """Instance role (design 6): read artifacts and models, write models and run evidence."""
        role.add_to_principal_policy(iam.PolicyStatement(actions=["s3:GetObject"], resources=self._obj(
            "artifacts/*", "models/*", "runs/*/cells/*")))
        role.add_to_principal_policy(iam.PolicyStatement(actions=["s3:PutObject"], resources=self._obj(
            "models/*", f"hosts/{target_id}/*", "runs/*/bringup/*", "runs/*/amx/*", "runs/*/hosts/*")))
        role.add_to_principal_policy(iam.PolicyStatement(actions=["s3:ListBucket"], resources=[self.bucket.bucket_arn],
                                                         conditions={"StringLike": {"s3:prefix": [
                                                             "artifacts/*", "models/*"]}}))
        role.add_to_principal_policy(iam.PolicyStatement(actions=["cloudwatch:PutMetricData"], resources=["*"],
                                                         conditions={"StringEquals": {
                                                             "cloudwatch:namespace": "AmxGpuBench"}}))
        self.hf_secret.grant_read(role)
        for a in self.image_assets.values():
            a.repository.grant_pull(role)
        if self.image_assets:
            # The registry login token: GetAuthorizationToken takes only "*" (it names no repository); which
            # repositories the host may then pull from is the grant_pull above, per image asset.
            role.add_to_principal_policy(iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"]))

