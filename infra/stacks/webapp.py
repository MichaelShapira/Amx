"""AmxBench-WebApp: the benchmark's web application on AWS, behind Amazon Cognito.

Shape (why each piece):

* ECS Fargate runs the container from webui/Dockerfile in the bench VPC's private subnet (no public
  IP), one always-on task (the API keeps AI-proposal jobs and caches in process), with a task role
  so the app never holds a credential. An internal Network Load Balancer in the same subnet is the
  only thing that can reach the task's port, and the only thing that can reach the load balancer
  is CloudFront, through a VPC origin (CloudFront's origin-facing addresses, inside the VPC). The
  CloudFront distribution is the single public entry: HTTPS on its own domain with a managed
  certificate, HTTP redirected. Public on purpose: the users are people in a browser anywhere;
  what gates them is the user pool, not the network. (AWS App Runner would have been simpler and
  was the first version; this account cannot create App Runner services.)
* Amazon Cognito user pool: nobody can sign up (admin-created users only, the deploy script
  creates the one user it is given), email sign-in, strong password policy, MFA offered, account
  recovery by email. The app client is public (PKCE, no secret), authorization-code only, with
  the service URL as its sole callback. The hosted login (managed login, Cognito domain) is the
  only place a password is ever typed.
* Authorization: any pool user may read; the `operators` group may start, stop and delete runs
  (webui/api/auth.py). The deploy script adds the initial user to the group.
* Least privilege for the service's role: the bench bucket's run prefixes and read-only data
  prefixes, the live model catalog and the prompt pools the Setup page writes, the one table, the
  run state machine, EC2 terminate only on instances tagged bench:managed, the pre-check
  function's configuration, Claude and OpenAI models on Bedrock, the AgentCore web-search tool,
  CloudWatch metric reads (the hosts' utilization),
  and the Hugging Face token secret (write, describe, and read for the onboarding job only).
* AWS WAF on the distribution: AWS IP reputation list, the common rule set and a per-IP rate limit.

Order of creation: pool and distribution first (neither depends on the other), then the app
client (its callback is the distribution's domain), then the task definition (it carries the
client id), then the service. No cycle, so the client id is a plain environment value.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudfront as cf
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_wafv2 as wafv2
from aws_cdk import custom_resources as cr
from constructs import Construct

ROOT = Path(__file__).resolve().parents[2]
SERVICE_NAME = "amx-gpu-bench-webapp"
OPERATOR_GROUP = "operators"
PORT = 8080
CLOUDFRONT_ORIGIN_FACING = "com.amazonaws.global.cloudfront.origin-facing"
# Bedrock: the models Setup -> Reasoning model may offer (webui/api/settings.py: Anthropic Claude and
# OpenAI only), by foundation model and by us. inference profile (which routes to the model in any
# US region). The list must stay in step with settings.ALLOWED_PROVIDERS.
BEDROCK_MODEL_ARNS = ["arn:aws:bedrock:*::foundation-model/anthropic.claude-*",
                      "arn:aws:bedrock:*::foundation-model/openai.*"]
BEDROCK_PROFILE_PATTERNS = ["us.anthropic.claude-*", "us.openai.*"]
# The AgentCore built-in web search tool (bedrock-agentcore:InvokeWebSearch), an AWS-owned resource.
WEB_SEARCH_TOOL_ARN = "arn:aws:bedrock-agentcore:{region}:aws:tool/web-search.v1"
PLACEHOLDER_IMAGE = "public.ecr.aws/docker/library/python:3.12-slim"   # synth without building (tests)


class WebAppStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, core, build_images: bool = True,
                 harness_commit: str = "unknown", **kw):
        super().__init__(scope, cid, **kw)
        self.core = core

        # ---------------------------------------------------------------- user pool
        self.pool = cognito.UserPool(
            self, "Users", user_pool_name="amx-gpu-bench",
            self_sign_up_enabled=False,                       # admin-created accounts only
            sign_in_aliases=cognito.SignInAliases(email=True),
            sign_in_case_sensitive=False,
            auto_verify=cognito.AutoVerifiedAttrs(email=True),
            standard_attributes=cognito.StandardAttributes(
                email=cognito.StandardAttribute(required=True, mutable=True)),
            password_policy=cognito.PasswordPolicy(min_length=8, require_lowercase=True, require_uppercase=True,
                                                   require_digits=True, require_symbols=True,
                                                   temp_password_validity=Duration.days(3)),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
            mfa=cognito.Mfa.OPTIONAL, mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            feature_plan=cognito.FeaturePlan.ESSENTIALS,
            email=cognito.UserPoolEmail.with_cognito(),
            deletion_protection=True, removal_policy=RemovalPolicy.RETAIN)
        cognito.CfnUserPoolGroup(self, "Operators", user_pool_id=self.pool.user_pool_id, group_name=OPERATOR_GROUP,
                                 description="May start, stop and delete benchmark runs and call Bedrock. "
                                             "Other users read results only.")
        self.domain = self.pool.add_domain(
            "HostedLogin", cognito_domain=cognito.CognitoDomainOptions(domain_prefix=f"amx-gpu-bench-{self.account}"),
            managed_login_version=cognito.ManagedLoginVersion.NEWER_MANAGED_LOGIN)

        # ---------------------------------------------------------------- network: task <- NLB <- CloudFront
        vpc = core.network.vpc
        sg_lb = ec2.SecurityGroup(self, "SgWebLb", vpc=vpc, allow_all_outbound=False,
                                  description="web app load balancer: HTTP from the CloudFront origin-facing addresses only")
        sg_lb.add_ingress_rule(ec2.Peer.prefix_list(self._managed_prefix_list(CLOUDFRONT_ORIGIN_FACING)),
                               ec2.Port.tcp(80), "CloudFront VPC origin")
        sg_task = ec2.SecurityGroup(self, "SgWebTask", vpc=vpc, allow_all_outbound=False,
                                    description="web app task: its port from the load balancer only; HTTPS out")
        sg_task.add_ingress_rule(sg_lb, ec2.Port.tcp(PORT), "app port from the load balancer")
        sg_task.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "AWS APIs, ECR, Bedrock, web search via NAT")
        sg_lb.add_egress_rule(sg_task, ec2.Port.tcp(PORT), "to the task")

        self.lb = elb.NetworkLoadBalancer(self, "Lb", vpc=vpc, internet_facing=False, security_groups=[sg_lb],
                                          vpc_subnets=core.network.private_subnets, cross_zone_enabled=False)
        self.distribution = self._distribution()
        url = f"https://{self.distribution.distribution_domain_name}"

        # ---------------------------------------------------------------- app client (callback = the distribution)
        self.client = self.pool.add_client(
            "Web", user_pool_client_name="amx-gpu-bench-web", generate_secret=False,
            auth_flows=cognito.AuthFlow(user_srp=False, user_password=False, admin_user_password=False, custom=False),
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True, implicit_code_grant=False, client_credentials=False),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
                callback_urls=[f"{url}/callback"], logout_urls=[f"{url}/"]),
            supported_identity_providers=[cognito.UserPoolClientIdentityProvider.COGNITO],
            prevent_user_existence_errors=True, enable_token_revocation=True,
            access_token_validity=Duration.hours(1), id_token_validity=Duration.hours(1),
            refresh_token_validity=Duration.hours(12))
        cognito.CfnManagedLoginBranding(self, "Branding", user_pool_id=self.pool.user_pool_id,
                                        client_id=self.client.user_pool_client_id, use_cognito_provided_values=True)

        # ---------------------------------------------------------------- the task and its role
        task_role = iam.Role(self, "TaskRole", assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
                             description="What the benchmark web app may do on behalf of signed-in users")
        self._grant(task_role)
        self.cluster = ecs.Cluster(self, "Cluster", vpc=vpc, cluster_name=SERVICE_NAME,
                                   container_insights_v2=ecs.ContainerInsights.ENABLED)
        self.task = ecs.FargateTaskDefinition(
            self, "Task", cpu=1024, memory_limit_mib=2048, task_role=task_role,
            runtime_platform=ecs.RuntimePlatform(cpu_architecture=ecs.CpuArchitecture.X86_64,
                                                 operating_system_family=ecs.OperatingSystemFamily.LINUX))
        self.asset = None
        if build_images:
            self.asset = ecr_assets.DockerImageAsset(self, "Image", directory=str(ROOT), file="webui/Dockerfile",
                                                     platform=ecr_assets.Platform.LINUX_AMD64)
            image = ecs.ContainerImage.from_docker_image_asset(self.asset)
        else:
            image = ecs.ContainerImage.from_registry(PLACEHOLDER_IMAGE)
        self.task.add_container(
            "web", image=image, environment=self._environment(harness_commit),
            port_mappings=[ecs.PortMapping(container_port=PORT, protocol=ecs.Protocol.TCP)],
            readonly_root_filesystem=False,                   # the data sync writes under /app/prompts and /app/data
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="web", log_group=logs.LogGroup(self, "Logs", log_group_name=f"/ecs/{SERVICE_NAME}",
                                                             retention=logs.RetentionDays.ONE_MONTH,
                                                             removal_policy=RemovalPolicy.DESTROY)),
            health_check=ecs.HealthCheck(command=["CMD-SHELL", f"python -c \"import urllib.request,sys; "
                                                  f"sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:{PORT}/api/health', "
                                                  f"timeout=3).status == 200 else 1)\""],
                                         interval=Duration.seconds(30), timeout=Duration.seconds(5), retries=3,
                                         start_period=Duration.seconds(120)))
        self.service = ecs.FargateService(
            self, "Service", cluster=self.cluster, service_name=SERVICE_NAME, task_definition=self.task,
            desired_count=1, min_healthy_percent=0, max_healthy_percent=100,     # one task; replaced in place
            assign_public_ip=False, vpc_subnets=core.network.private_subnets, security_groups=[sg_task],
            circuit_breaker=ecs.DeploymentCircuitBreaker(enable=True, rollback=True),
            enable_execute_command=False, health_check_grace_period=Duration.seconds(180),
            # Tags on the service do not reach its tasks by themselves; the tasks are what the bill meters.
            propagate_tags=ecs.PropagatedTagSource.SERVICE, enable_ecs_managed_tags=True)
        listener = self.lb.add_listener("Http", port=80, protocol=elb.Protocol.TCP)
        listener.add_targets("Web", port=PORT, protocol=elb.Protocol.TCP, targets=[self.service],
                             deregistration_delay=Duration.seconds(30), preserve_client_ip=False,
                             health_check=elb.HealthCheck(protocol=elb.Protocol.HTTP, path="/api/health",
                                                          interval=Duration.seconds(10), healthy_threshold_count=2,
                                                          unhealthy_threshold_count=3))

        for k, v in {"ServiceUrl": url, "UserPoolId": self.pool.user_pool_id,
                     "UserPoolClientId": self.client.user_pool_client_id, "HostedLoginDomain": self.domain.base_url(),
                     "OperatorGroup": OPERATOR_GROUP, "DistributionId": self.distribution.distribution_id,
                     "LogGroup": f"/ecs/{SERVICE_NAME}"}.items():
            CfnOutput(self, k, value=v)

    # -------------------------------------------------------------------- helpers

    def _environment(self, harness_commit: str) -> dict[str, str]:
        core = self.core
        outputs = {"BucketName": core.bucket.bucket_name, "TableName": core.table.table_name,
                   "RunStateMachineArn": core.run_sm.state_machine_arn,
                   "LevelStateMachineArn": core.level_sm.state_machine_arn}
        return {
            # In-container bind; only the load balancer's security group reaches the port, and the
            # API refuses this bind without the pool.
            "AMXBENCH_BIND": "0.0.0.0", "PORT": str(PORT), "AMXBENCH_DATA_SYNC": "1", "AMXBENCH_CREDS_FILE": "",  # noqa: S104  # nosec B104
            "AMXBENCH_OUTPUTS": self.to_json_string(outputs),
            "AMXBENCH_COGNITO_USER_POOL_ID": self.pool.user_pool_id,
            "AMXBENCH_COGNITO_REGION": self.region,
            "AMXBENCH_COGNITO_DOMAIN": self.domain.base_url(),
            "AMXBENCH_COGNITO_CLIENT_ID": self.client.user_pool_client_id,
            "AMXBENCH_HF_SECRET_ARN": core.hf_secret.secret_arn,      # Setup -> Hugging Face token writes here
            "HARNESS_COMMIT": harness_commit,
        }

    def _distribution(self) -> cf.Distribution:
        """CloudFront in front of the internal load balancer through a VPC origin. Nothing is cached
        except the content-hashed /assets; every request forwards the viewer's headers (the
        Authorization bearer token among them) and query string to the app."""
        acl = wafv2.CfnWebACL(
            self, "Waf", name=f"{SERVICE_NAME}-waf", scope="CLOUDFRONT", default_action={"allow": {}},
            visibility_config=self._vis(f"{SERVICE_NAME}-waf"),
            rules=[
                {"name": "RateLimitPerIp", "priority": 1, "action": {"block": {}},
                 "statement": {"rateBasedStatement": {"limit": 1000, "aggregateKeyType": "IP"}},
                 "visibilityConfig": self._vis("RateLimitPerIp")},
                {"name": "AWSManagedRulesAmazonIpReputationList", "priority": 2, "overrideAction": {"none": {}},
                 "statement": {"managedRuleGroupStatement": {"vendorName": "AWS",
                                                             "name": "AWSManagedRulesAmazonIpReputationList"}},
                 "visibilityConfig": self._vis("IpReputation")},
                {"name": "AWSManagedRulesCommonRuleSet", "priority": 3, "overrideAction": {"none": {}},
                 "statement": {"managedRuleGroupStatement": {"vendorName": "AWS", "name": "AWSManagedRulesCommonRuleSet"}},
                 "visibilityConfig": self._vis("CommonRuleSet")},
            ])
        origin = origins.VpcOrigin.with_network_load_balancer(
            self.lb, vpc_origin_name=SERVICE_NAME, http_port=80, protocol_policy=cf.OriginProtocolPolicy.HTTP_ONLY,
            read_timeout=Duration.seconds(60), keepalive_timeout=Duration.seconds(60))
        app_behavior = dict(origin=origin, viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                            allowed_methods=cf.AllowedMethods.ALLOW_ALL, cached_methods=cf.CachedMethods.CACHE_GET_HEAD,
                            cache_policy=cf.CachePolicy.CACHING_DISABLED,
                            origin_request_policy=cf.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER, compress=True)
        return cf.Distribution(
            self, "Distribution", comment="GPU vs AMX CPU benchmark web app (Cognito-gated)",
            default_behavior=cf.BehaviorOptions(**app_behavior),
            additional_behaviors={"/assets/*": cf.BehaviorOptions(
                origin=origin, viewer_protocol_policy=cf.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                allowed_methods=cf.AllowedMethods.ALLOW_GET_HEAD, cache_policy=cf.CachePolicy.CACHING_OPTIMIZED,
                compress=True)},
            web_acl_id=acl.attr_arn, http_version=cf.HttpVersion.HTTP2_AND_3,
            minimum_protocol_version=cf.SecurityPolicyProtocol.TLS_V1_2_2021,
            price_class=cf.PriceClass.PRICE_CLASS_100)

    def _managed_prefix_list(self, name: str) -> str:
        """The id of an AWS-managed prefix list (CloudFront's origin-facing addresses) at deploy time."""
        res = cr.AwsCustomResource(
            self, "CloudFrontPrefixList",
            on_update=cr.AwsSdkCall(service="EC2", action="describeManagedPrefixLists",
                                    parameters={"Filters": [{"Name": "prefix-list-name", "Values": [name]}]},
                                    physical_resource_id=cr.PhysicalResourceId.of(f"pl-{name}"),
                                    output_paths=["PrefixLists.0.PrefixListId"]),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE),
            install_latest_aws_sdk=False)
        return res.get_response_field("PrefixLists.0.PrefixListId")

    def _grant(self, role: iam.Role) -> None:
        core, acct, region = self.core, self.account, self.region
        bucket = core.bucket
        read_prefixes = ["runs/*", "artifacts/prompts/*", "datasets/*", "artifacts/dspm/*", "config/*", "hosts/*"]
        role.add_to_policy(iam.PolicyStatement(
            sid="ListRunAndDataPrefixes", actions=["s3:ListBucket"], resources=[bucket.bucket_arn],
            conditions={"StringLike": {"s3:prefix": read_prefixes}}))
        role.add_to_policy(iam.PolicyStatement(
            sid="ReadRunsAndData", actions=["s3:GetObject"],
            resources=[bucket.arn_for_objects(p) for p in read_prefixes]))
        role.add_to_policy(iam.PolicyStatement(
            sid="WriteRunConfig", actions=["s3:PutObject"], resources=[bucket.arn_for_objects("runs/*/run.yaml")]))
        # Setup -> Models: the live catalog object and the prompt pools of a model a user adds. No delete:
        # removing a model rewrites the catalog and leaves its pools in place.
        role.add_to_policy(iam.PolicyStatement(
            sid="WriteCatalogAndPools", actions=["s3:PutObject"],
            resources=[bucket.arn_for_objects(p) for p in
                       ("config/models.yaml", "artifacts/prompts/*", "artifacts/dspm/v1/requests/*")]))
        # Setup -> Hugging Face token: the app writes the value and reports when it last changed; it reads
        # the value only inside the onboarding job (a gated model's tokenizer download) and never returns it.
        role.add_to_policy(iam.PolicyStatement(
            sid="HfTokenSecret", actions=["secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret",
                                          "secretsmanager:GetSecretValue"],
            resources=[core.hf_secret.secret_arn]))
        role.add_to_policy(iam.PolicyStatement(
            sid="DeleteRun", actions=["s3:DeleteObject"], resources=[bucket.arn_for_objects("runs/*")]))
        # Scan stays: the Runs page lists every run's META row, which a single-table design without a
        # secondary index can only reach by scan (the table is the app's own data, not a shared store).
        role.add_to_policy(iam.PolicyStatement(
            sid="RunTable", resources=[core.table.table_arn],
            actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
                     "dynamodb:Query", "dynamodb:Scan"]))
        role.add_to_policy(iam.PolicyStatement(
            sid="StartAndListRuns", actions=["states:StartExecution", "states:ListExecutions"],
            resources=[core.run_sm.state_machine_arn]))
        role.add_to_policy(iam.PolicyStatement(
            sid="InspectAndStopRuns", actions=["states:DescribeExecution", "states:StopExecution"],
            resources=[f"arn:aws:states:{region}:{acct}:execution:{core.run_sm.state_machine_name}:*"]))
        # ec2:Describe* and bedrock:List* support no resource ARN or tag condition (IAM reference: these
        # actions take only "*"); the role sees instance metadata and model names, nothing it can act on
        # beyond the conditioned TerminateInstances and the named Claude models below.
        role.add_to_policy(iam.PolicyStatement(sid="SeeInstances", actions=["ec2:DescribeInstances"], resources=["*"]))
        role.add_to_policy(iam.PolicyStatement(
            sid="TerminateOnlyBenchInstances", actions=["ec2:TerminateInstances"],
            resources=[f"arn:aws:ec2:{region}:{acct}:instance/*"],
            conditions={"StringEquals": {"aws:ResourceTag/bench:managed": "true"}}))
        role.add_to_policy(iam.PolicyStatement(
            sid="ReadPreflightVersion", actions=["lambda:GetFunctionConfiguration"],
            resources=[core.preflight_fn.function_arn]))
        role.add_to_policy(iam.PolicyStatement(
            sid="ReasoningModelsOnBedrock", actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=[*BEDROCK_MODEL_ARNS,
                       *(f"arn:aws:bedrock:{region}:{acct}:inference-profile/{p}" for p in BEDROCK_PROFILE_PATTERNS)]))
        role.add_to_policy(iam.PolicyStatement(
            sid="ListBedrockModels", actions=["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
            resources=["*"]))
        # The Results page reads the test hosts' CPU/GPU/memory utilization over a run (the agent's
        # AmxGpuBench namespace). GetMetricData supports no resource ARN or condition key: read-only, "*".
        role.add_to_policy(iam.PolicyStatement(sid="ReadHostUtilization", actions=["cloudwatch:GetMetricData"], resources=["*"]))
        role.add_to_policy(iam.PolicyStatement(
            sid="WebSearchTool", actions=["bedrock-agentcore:InvokeWebSearch"],
            resources=[WEB_SEARCH_TOOL_ARN.format(region=region)]))

    def _vis(self, metric: str) -> dict:
        return {"sampledRequestsEnabled": True, "cloudWatchMetricsEnabled": True, "metricName": metric}


__all__ = ["WebAppStack", "OPERATOR_GROUP", "SERVICE_NAME"]
