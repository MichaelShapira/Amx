"""AmxBench-WebApp: the benchmark's web application on AWS, behind Amazon Cognito.

Shape (why each piece):

* AWS App Runner runs the container from webui/Dockerfile: HTTPS on a platform domain with a
  managed certificate, one always-on instance (the API keeps AI-proposal jobs and caches in
  process), the platform publishes the port, and it carries an instance role, so the app never
  holds a credential. Ingress is public on purpose: the users are people in a browser anywhere;
  what gates them is the user pool, not the network.
* Amazon Cognito user pool: nobody can sign up (admin-created users only, the deploy script
  creates the one user it is given), email sign-in, strong password policy, MFA offered, account
  recovery by email. The app client is public (PKCE, no secret), authorization-code only, with
  the service URL as its sole callback. The hosted login (managed login, Cognito domain) is the
  only place a password is ever typed.
* Authorization: any pool user may read; the `operators` group may start, stop and delete runs
  (webui/api/auth.py). The deploy script adds the initial user to the group.
* Least privilege for the service's role: the bench bucket's run prefixes and read-only data
  prefixes, the one table, the run state machine, EC2 terminate only on instances tagged
  bench:managed, the pre-check function's configuration, Claude on Bedrock, one SSM parameter.
* AWS WAF in front: AWS IP reputation list, the common rule set and a per-IP rate limit.

The client's callback URL is the service URL, which exists only after the service, and the
service needs the client id: the id goes into an SSM parameter the app reads at start, so the
dependency is pool -> service -> client -> parameter, with no cycle.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apprunner as apprunner
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_ssm as ssm
from aws_cdk import aws_wafv2 as wafv2
from constructs import Construct

ROOT = Path(__file__).resolve().parents[2]
SERVICE_NAME = "amx-gpu-bench-webapp"
OPERATOR_GROUP = "operators"
CLIENT_ID_PARAM = "/amx-gpu-bench/webapp/cognito-client-id"
PORT = "8080"
# Claude on Bedrock: the models webui/api/exec_summary.py and ai_propose.py can pick (newest Opus
# via an inference profile, which routes to the foundation model in any US region).
BEDROCK_MODEL_ARNS = ["arn:aws:bedrock:*::foundation-model/anthropic.claude-*"]
PLACEHOLDER_IMAGE = "public.ecr.aws/aws-containers/hello-app-runner:latest"   # synth without building


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

        # ---------------------------------------------------------------- roles
        instance_role = iam.Role(self, "InstanceRole", assumed_by=iam.ServicePrincipal("tasks.apprunner.amazonaws.com"),
                                 description="What the benchmark web app may do on behalf of signed-in users")
        self._grant(instance_role)

        # ---------------------------------------------------------------- service
        self.asset = None
        if build_images:
            self.asset = ecr_assets.DockerImageAsset(self, "Image", directory=str(ROOT), file="webui/Dockerfile",
                                                     platform=ecr_assets.Platform.LINUX_AMD64)
            access_role = iam.Role(self, "EcrAccessRole",
                                   assumed_by=iam.ServicePrincipal("build.apprunner.amazonaws.com"),
                                   description="Lets App Runner pull the web app image from the CDK asset repository")
            self.asset.repository.grant_pull(access_role)
            image = apprunner.CfnService.ImageRepositoryProperty(
                image_identifier=self.asset.image_uri, image_repository_type="ECR",
                image_configuration=self._image_configuration(harness_commit))
            auth_cfg = apprunner.CfnService.AuthenticationConfigurationProperty(access_role_arn=access_role.role_arn)
        else:
            image = apprunner.CfnService.ImageRepositoryProperty(
                image_identifier=PLACEHOLDER_IMAGE, image_repository_type="ECR_PUBLIC",
                image_configuration=self._image_configuration(harness_commit))
            auth_cfg = None

        scaling = apprunner.CfnAutoScalingConfiguration(
            self, "Scaling", auto_scaling_configuration_name=f"{SERVICE_NAME}-single", min_size=1, max_size=1,
            max_concurrency=100)
        self.service = apprunner.CfnService(
            self, "Service", service_name=SERVICE_NAME,
            source_configuration=apprunner.CfnService.SourceConfigurationProperty(
                auto_deployments_enabled=False, authentication_configuration=auth_cfg, image_repository=image),
            instance_configuration=apprunner.CfnService.InstanceConfigurationProperty(
                cpu="1 vCPU", memory="2 GB", instance_role_arn=instance_role.role_arn),
            health_check_configuration=apprunner.CfnService.HealthCheckConfigurationProperty(
                protocol="HTTP", path="/api/health", interval=10, timeout=5, healthy_threshold=1, unhealthy_threshold=5),
            network_configuration=apprunner.CfnService.NetworkConfigurationProperty(
                ingress_configuration=apprunner.CfnService.IngressConfigurationProperty(is_publicly_accessible=True),
                egress_configuration=apprunner.CfnService.EgressConfigurationProperty(egress_type="DEFAULT")),
            auto_scaling_configuration_arn=scaling.attr_auto_scaling_configuration_arn)
        url = f"https://{self.service.attr_service_url}"

        # ---------------------------------------------------------------- app client (after the service: its URL)
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
        self.client_param = ssm.StringParameter(self, "ClientIdParam", parameter_name=CLIENT_ID_PARAM,
                                                string_value=self.client.user_pool_client_id,
                                                description="App client id of the benchmark web app (read by the app at start)")
        self.client_param.grant_read(instance_role)

        # ---------------------------------------------------------------- WAF
        acl = wafv2.CfnWebACL(
            self, "Waf", name=f"{SERVICE_NAME}-waf", scope="REGIONAL", default_action={"allow": {}},
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
        wafv2.CfnWebACLAssociation(self, "WafAssociation", resource_arn=self.service.attr_service_arn,
                                   web_acl_arn=acl.attr_arn)

        for k, v in {"ServiceUrl": url, "UserPoolId": self.pool.user_pool_id,
                     "UserPoolClientId": self.client.user_pool_client_id, "HostedLoginDomain": self.domain.base_url(),
                     "OperatorGroup": OPERATOR_GROUP, "ClientIdParameter": CLIENT_ID_PARAM}.items():
            CfnOutput(self, k, value=v)

    # -------------------------------------------------------------------- helpers

    def _image_configuration(self, harness_commit: str) -> apprunner.CfnService.ImageConfigurationProperty:
        core = self.core
        outputs = {"BucketName": core.bucket.bucket_name, "TableName": core.table.table_name,
                   "RunStateMachineArn": core.run_sm.state_machine_arn,
                   "LevelStateMachineArn": core.level_sm.state_machine_arn}
        env = {
            # In-container bind; App Runner publishes the port and the API refuses this bind without the pool.
            "AMXBENCH_BIND": "0.0.0.0", "PORT": PORT, "AMXBENCH_DATA_SYNC": "1", "AMXBENCH_CREDS_FILE": "",  # noqa: S104
            "AMXBENCH_OUTPUTS": self.to_json_string(outputs),
            "AMXBENCH_COGNITO_USER_POOL_ID": self.pool.user_pool_id,
            "AMXBENCH_COGNITO_REGION": self.region,
            "AMXBENCH_COGNITO_DOMAIN": self.domain.base_url(),
            "AMXBENCH_COGNITO_CLIENT_ID_PARAM": CLIENT_ID_PARAM,
            "HARNESS_COMMIT": harness_commit,
        }
        return apprunner.CfnService.ImageConfigurationProperty(
            port=PORT, runtime_environment_variables=[
                apprunner.CfnService.KeyValuePairProperty(name=k, value=v) for k, v in env.items()])

    def _grant(self, role: iam.Role) -> None:
        core, acct, region = self.core, self.account, self.region
        bucket = core.bucket
        read_prefixes = ["runs/*", "artifacts/prompts/*", "datasets/*", "artifacts/dspm/*"]
        role.add_to_policy(iam.PolicyStatement(
            sid="ListRunAndDataPrefixes", actions=["s3:ListBucket"], resources=[bucket.bucket_arn],
            conditions={"StringLike": {"s3:prefix": read_prefixes}}))
        role.add_to_policy(iam.PolicyStatement(
            sid="ReadRunsAndData", actions=["s3:GetObject"],
            resources=[bucket.arn_for_objects(p) for p in read_prefixes]))
        role.add_to_policy(iam.PolicyStatement(
            sid="WriteRunConfig", actions=["s3:PutObject"], resources=[bucket.arn_for_objects("runs/*/run.yaml")]))
        role.add_to_policy(iam.PolicyStatement(
            sid="DeleteRun", actions=["s3:DeleteObject"], resources=[bucket.arn_for_objects("runs/*")]))
        role.add_to_policy(iam.PolicyStatement(
            sid="RunTable", resources=[core.table.table_arn],
            actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
                     "dynamodb:Query", "dynamodb:Scan", "dynamodb:BatchWriteItem"]))
        role.add_to_policy(iam.PolicyStatement(
            sid="StartAndListRuns", actions=["states:StartExecution", "states:ListExecutions"],
            resources=[core.run_sm.state_machine_arn]))
        role.add_to_policy(iam.PolicyStatement(
            sid="InspectAndStopRuns", actions=["states:DescribeExecution", "states:StopExecution"],
            resources=[f"arn:aws:states:{region}:{acct}:execution:{core.run_sm.state_machine_name}:*"]))
        role.add_to_policy(iam.PolicyStatement(sid="SeeInstances", actions=["ec2:DescribeInstances"], resources=["*"]))
        role.add_to_policy(iam.PolicyStatement(
            sid="TerminateOnlyBenchInstances", actions=["ec2:TerminateInstances"],
            resources=[f"arn:aws:ec2:{region}:{acct}:instance/*"],
            conditions={"StringEquals": {"aws:ResourceTag/bench:managed": "true"}}))
        role.add_to_policy(iam.PolicyStatement(
            sid="ReadPreflightVersion", actions=["lambda:GetFunctionConfiguration"],
            resources=[core.preflight_fn.function_arn]))
        role.add_to_policy(iam.PolicyStatement(
            sid="ClaudeOnBedrock", actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
            resources=[*BEDROCK_MODEL_ARNS, f"arn:aws:bedrock:{region}:{acct}:inference-profile/us.anthropic.claude-*"]))
        role.add_to_policy(iam.PolicyStatement(
            sid="ListBedrockModels", actions=["bedrock:ListFoundationModels", "bedrock:ListInferenceProfiles"],
            resources=["*"]))

    def _vis(self, metric: str) -> dict:
        return {"sampledRequestsEnabled": True, "cloudWatchMetricsEnabled": True, "metricName": metric}


def service_env_for_tests(stack: WebAppStack) -> dict:
    """The environment the service passes to the container, resolved (tests)."""
    svc = stack.service
    cfg = svc.source_configuration.image_repository.image_configuration
    return {kv.name: stack.resolve(kv.value) for kv in cfg.runtime_environment_variables}


__all__ = ["WebAppStack", "service_env_for_tests", "CLIENT_ID_PARAM", "OPERATOR_GROUP", "SERVICE_NAME"]
