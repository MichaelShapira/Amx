"""CoreStack (design 3, 6, 9, 10): bucket, table, images, Lambdas with least-privilege
roles, the run and level state machines, the reaper schedule and the host scripts."""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as ddb
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
from constructs import Construct

from infra.stacks.bundling import lambda_code
from infra.stacks.network import NetworkStack

ROOT = Path(__file__).resolve().parents[2]
HF_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:346399954218:secret:amx-gpu-bench/hf-token-naqygX"  # noqa: S105 - an ARN, not a value
WORKER_RESERVED = 80          # highest level (64) plus the accuracy level's 16 headroom
IMAGES = {
    "thin-torch-cpu": ("servers/thin/Dockerfile.torch", {
        "BASE_IMAGE": "public.ecr.aws/deep-learning-containers/pytorch:2.14.0-cpu-amzn2023@sha256:"
                      "e9aa07c11afc5f7eff183782cf3dde397012e2f08ab700dffd2828203688c159"}),
    "thin-torch-cuda": ("servers/thin/Dockerfile.torch", {
        "BASE_IMAGE": "public.ecr.aws/deep-learning-containers/pytorch:2.14.0-cu133-amzn2023@sha256:"
                      "f9e5a7febd6ef9104deb79941b58c7b054091a29f2b661fe9c2ea076886cb7be"}),
    "thin-openvino": ("servers/thin/Dockerfile.openvino", {}),
    "export": ("servers/export/Dockerfile", {}),
}
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
        self.hf_secret = sm.Secret.from_secret_complete_arn(self, "HfToken", HF_SECRET_ARN)

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
        control = fn("target_control", "lambdas.target_control.handler", memory=256)
        reaper = fn("reaper", "lambdas.reaper.handler", memory=256, timeout_min=5,
                    env={"IDLE_MINUTES": "30", "MAX_UPTIME_HOURS": "8"})
        preflight = fn("preflight", "lambdas.preflight.handler", memory=1024, timeout_min=5,
                       env={"WORKER_FUNCTION_NAME": "amx-gpu-bench-worker", "HF_TOKEN_SECRET_ARN": HF_SECRET_ARN,
                            "ECR_REPOSITORY_URI": self.ecr_repo_uri, "HARNESS_COMMIT": harness_commit})

        self._grant(worker, warmup, quality, init_level, plan_wave, wave_check, perf, amx_probe, membw, agg_level,
                    agg_run, control, reaper, preflight)
        self.level_sm = self._level_state_machine(init_level, warmup, perf, plan_wave, worker, wave_check, agg_level)
        self.run_sm = self._run_state_machine(preflight, control, amx_probe, membw, quality, agg_run)
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
        ssm_send = iam.PolicyStatement(
            actions=["ssm:SendCommand"],
            resources=[f"arn:aws:ssm:{self.region}::document/AWS-RunShellScript"])
        ssm_send_inst = iam.PolicyStatement(
            actions=["ssm:SendCommand"], resources=[f"arn:aws:ec2:{self.region}:{self.account}:instance/*"],
            conditions={"StringEquals": {"ssm:resourceTag/bench:managed": "true"}})
        ssm_read = iam.PolicyStatement(actions=["ssm:GetCommandInvocation", "ssm:DescribeInstanceInformation"],
                                       resources=["*"])
        for f in (perf, amx_probe, membw, control):
            for st in (ssm_send, ssm_send_inst, ssm_read):
                f.add_to_role_policy(st)
        amx_probe.add_to_role_policy(s3_get("runs/*/amx/*"))
        amx_probe.add_to_role_policy(s3_put("runs/*/amx/*"))
        ec2_tagged = {"StringEquals": {"aws:ResourceTag/bench:managed": "true"}}
        for f in (control, reaper):
            f.add_to_role_policy(iam.PolicyStatement(actions=["ec2:DescribeInstances"], resources=["*"]))
            f.add_to_role_policy(iam.PolicyStatement(
                actions=["ec2:StopInstances"], resources=[f"arn:aws:ec2:{self.region}:{self.account}:instance/*"],
                conditions=ec2_tagged))
            f.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:GetItem", "dynamodb:UpdateItem"],
                                                     resources=[t.table_arn]))
        control.add_to_role_policy(iam.PolicyStatement(
            actions=["ec2:StartInstances"], resources=[f"arn:aws:ec2:{self.region}:{self.account}:instance/*"],
            conditions=ec2_tagged))
        control.add_to_role_policy(s3_get("runs/*"))
        control.add_to_role_policy(s3_put("runs/*/capacity/*", "runs/*/plan/*"))
        agg_level.add_to_role_policy(s3_list)
        agg_level.add_to_role_policy(s3_get("runs/*"))
        agg_level.add_to_role_policy(s3_put("runs/*/metrics/*", "runs/*/predictions/*"))
        agg_level.add_to_role_policy(iam.PolicyStatement(actions=["cloudwatch:GetMetricData"], resources=["*"]))
        agg_level.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        agg_run.add_to_role_policy(s3_list)
        agg_run.add_to_role_policy(s3_get("runs/*", "datasets/*"))
        agg_run.add_to_role_policy(s3_put("runs/*/report/*"))
        agg_run.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        preflight.add_to_role_policy(s3_list)
        preflight.add_to_role_policy(s3_get("runs/*", "hosts/*", "artifacts/*"))
        preflight.add_to_role_policy(s3_put("runs/*"))
        preflight.add_to_role_policy(iam.PolicyStatement(actions=["dynamodb:UpdateItem"], resources=[t.table_arn]))
        preflight.add_to_role_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"],
            resources=[f"arn:aws:ssm:{self.region}:{self.account}:parameter/amx-gpu-bench/*"]))
        preflight.add_to_role_policy(iam.PolicyStatement(
            actions=["lambda:GetFunctionConcurrency"], resources=[self.fns["worker"].function_arn]))
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
        fail = sfn.Fail(self, "LevelFailed", cause="a wave failed twice", error="WaveFailed")
        perf_stop = invoke("PerfStop", perf, {"action": "stop", "level.$": "$.level"},
                           result_selector={"counts.$": "$.Payload.counts"}, result_path="$.perf")
        no_perf = sfn.Pass(self, "NoPerf", result=sfn.Result.from_object({"counts": None}), result_path="$.perf")
        aggregate = invoke("AggregateLevel", agg_level, {"level.$": "$.level", "perf.$": "$.perf"},
                           result_selector={"n_ok.$": "$.Payload.n_ok"}, result_path="$.result")
        perf_start.add_catch(plan, result_path=P.DISCARD)
        perf_stop.add_catch(no_perf, result_path=P.DISCARD)
        measure.add_catch(sfn.Choice(self, "WaveFailedOnce").when(
            sfn.Condition.number_equals("$.retries", 0), retry_wave.next(plan)).otherwise(fail), result_path=P.DISCARD)
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
                                     "stop.$": "$.Payload.stop_targets_after_run"}, result_path="$.pf")
        # Per cell
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
        levels = sfn.DistributedMap(
            self, "Levels", item_reader=sfn.S3JsonItemReader(bucket=self.bucket, key=P.string_at("$.cfg.r.levels_key")),
            max_concurrency=1, map_execution_type=sfn.StateMachineType.STANDARD, tolerated_failure_percentage=100,
            item_selector={"level.$": "$$.Map.Item.Value"}, result_path=P.DISCARD)
        levels.item_processor(tasks.StepFunctionsStartExecution(
            self, "RunLevel", state_machine=self.level_sm, integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            input=sfn.TaskInput.from_json_path_at("$"), result_selector={"ok": True}))
        cell_done = sfn.Pass(self, "CellDone")
        cell_failed = sfn.Pass(self, "CellFailed", parameters={"failed.$": "$.cell.cell_id", "error.$": "$.error"})
        after_mem = sfn.Choice(self, "NeedsQuality").when(sfn.Condition.boolean_equals("$.cell.quality", True),
                                                          qual.next(levels)).otherwise(levels)
        after_amx = sfn.Choice(self, "NeedsMemBw").when(sfn.Condition.boolean_equals("$.cell.membw", True),
                                                        mem.next(after_mem)).otherwise(after_mem)
        after_cfg = sfn.Choice(self, "NeedsAmx").when(sfn.Condition.boolean_equals("$.cell.amx", True),
                                                      amx.next(after_amx)).otherwise(after_amx)
        ready = sfn.Choice(self, "CellReady").when(sfn.Condition.boolean_equals("$.cfg.r.done", True), after_cfg) \
            .otherwise(cpoll)
        levels.next(cell_done)
        for st in (cstart, cpoll, amx, mem, qual, levels):
            st.add_catch(cell_failed, result_path="$.error")
        cells = sfn.Map(self, "Cells", items_path="$.target.cells", max_concurrency=1, result_path=P.DISCARD,
                        item_selector={"run_id.$": "$.run_id", "target_id.$": "$.target.target_id",
                                       "instance_id.$": "$.target.instance_id", "private_ip.$": "$.inst.private_ip",
                                       "cell.$": "$$.Map.Item.Value"})
        cells.item_processor(cstart.next(cpoll).next(ready))
        ensure = invoke("EnsureRunning", control, {"action": "ensure_running", "run_id.$": "$.run_id",
                                                    "target_id.$": "$.target.target_id",
                                                    "instance_id.$": "$.target.instance_id"},
                        result_selector={"private_ip.$": "$.Payload.private_ip"}, result_path="$.inst")
        target_failed = sfn.Pass(self, "TargetFailed")
        ensure.add_catch(target_failed, result_path="$.error")
        tmap = sfn.Map(self, "Targets", items_path="$.pf.targets", max_concurrency=0, result_path=P.DISCARD,
                       item_selector={"run_id.$": "$.pf.run_id", "target.$": "$$.Map.Item.Value"})
        tmap.item_processor(ensure.next(cells))
        agg = invoke("AggregateRun", agg_run, {"run_id.$": "$.pf.run_id"},
                     result_selector={"report_prefix.$": "$.Payload.report_prefix"}, result_path="$.report")
        stop_one = invoke("StopTarget", control, {"action": "stop", "target_id.$": "$.target_id",
                                                  "instance_id.$": "$.instance_id"}, result_path=P.DISCARD)
        stop_map = sfn.Map(self, "StopTargets", items_path="$.pf.targets", result_path=P.DISCARD,
                           item_selector={"target_id.$": "$$.Map.Item.Value.target_id",
                                          "instance_id.$": "$$.Map.Item.Value.instance_id"})
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
            role.add_to_principal_policy(iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"]))

