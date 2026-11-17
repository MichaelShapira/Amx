"""CDK assertion tests (task 15.4): network exposure, IMDSv2, S3 settings, tags, worker
concurrency and least-privilege statements."""

from __future__ import annotations

import json

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk.assertions import Match, Template  # noqa: E402

from infra.app import PROJECT_TAGS, build  # noqa: E402


@pytest.fixture(scope="module")
def templates():
    app = cdk.App(context={"az_name": "us-east-1a", "skip_bundling": "true", "build_images": "false",
                           "@aws-cdk/core:defaultCrossStackReferences": "strong"})
    stacks = build(app)
    return {k: Template.from_stack(v) for k, v in stacks.items()}


def _all(templates, rtype):
    for name, t in templates.items():
        for lid, r in t.find_resources(rtype).items():
            yield name, lid, r


def test_only_ingress_is_8000_from_workers(templates):
    net = templates["network"].to_json()["Resources"]
    ingress = [r for r in net.values() if r["Type"] == "AWS::EC2::SecurityGroupIngress"]
    assert len(ingress) == 1
    props = ingress[0]["Properties"]
    assert props["FromPort"] == 8000 == props["ToPort"] and "SourceSecurityGroupId" in props
    for _, _, sg in _all(templates, "AWS::EC2::SecurityGroup"):
        for rule in sg["Properties"].get("SecurityGroupIngress", []):
            assert rule.get("CidrIp") not in ("0.0.0.0/0",) and rule.get("CidrIpv6") != "::/0"
    blob = json.dumps({k: t.to_json() for k, t in templates.items()})
    assert '"CidrIp": "0.0.0.0/0", "Description": "HTTPS out' in blob or "HTTPS out via NAT" in blob


def test_launch_templates_private_imdsv2_encrypted_tagged(templates):
    """Instances are launched per run (On-Demand or Spot) from one launch template per target;
    the stacks themselves create no instance."""
    assert list(_all(templates, "AWS::EC2::Instance")) == []
    n = 0
    for name, _, lt in _all(templates, "AWS::EC2::LaunchTemplate"):
        n += 1
        d = lt["Properties"]["LaunchTemplateData"]
        assert "KeyName" not in d
        assert d["MetadataOptions"]["HttpTokens"] == "required"
        assert all(x.get("AssociatePublicIpAddress") is False and "SubnetId" in x for x in d["NetworkInterfaces"])
        bd = d["BlockDeviceMappings"][0]["Ebs"]
        assert bd["Encrypted"] is True and bd["VolumeType"] == "gp3"
        tags = {t["Key"]: t["Value"] for spec in d["TagSpecifications"] for t in spec["Tags"]
                if spec["ResourceType"] == "instance"}
        assert tags["bench:managed"] == "true" and tags["bench:target"] == name
        for k, v in PROJECT_TAGS.items():
            assert tags[k] == v
    assert n == 6


def test_control_launch_is_scoped(templates):
    core = templates["core"].to_json()["Resources"]
    stmts = [st for r in core.values() if r["Type"] == "AWS::IAM::Policy"
             for st in r["Properties"]["PolicyDocument"]["Statement"]]
    run = [st for st in stmts if "ec2:RunInstances" in (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])]
    assert len(run) == 1
    res = json.dumps(run[0]["Resource"])
    assert ":subnet/" in res and ":security-group/" in res and "launch-template/*" in res
    passrole = [st for st in stmts if st["Action"] == "iam:PassRole"]
    assert passrole and all(st["Condition"]["StringEquals"]["iam:PassedToService"] == "ec2.amazonaws.com"
                            and "role/amx-gpu-bench-target-*" in json.dumps(st["Resource"]) for st in passrole)
    tagging = [st for st in stmts if st["Action"] == "ec2:CreateTags"]
    assert tagging and all(st["Condition"]["StringEquals"]["ec2:CreateAction"] == "RunInstances" for st in tagging)


def test_public_subnet_never_maps_public_ips(templates):
    for _, _, sn in _all(templates, "AWS::EC2::Subnet"):
        assert sn["Properties"].get("MapPublicIpOnLaunch") in (False, None)


def test_bucket_private_tls_only_encrypted(templates):
    core = templates["core"]
    core.has_resource_properties("AWS::S3::Bucket", {
        "PublicAccessBlockConfiguration": {"BlockPublicAcls": True, "BlockPublicPolicy": True,
                                           "IgnorePublicAcls": True, "RestrictPublicBuckets": True},
        "BucketEncryption": {"ServerSideEncryptionConfiguration": [
            {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}})
    core.has_resource_properties("AWS::S3::BucketPolicy", {"PolicyDocument": {"Statement": Match.array_with([
        Match.object_like({"Effect": "Deny", "Condition": {"Bool": {"aws:SecureTransport": "false"}}})])}})
    for _, _, pol in _all(templates, "AWS::S3::BucketPolicy"):
        for st in pol["Properties"]["PolicyDocument"]["Statement"]:
            if st["Effect"] == "Allow":
                assert st.get("Principal") != "*" and st.get("Principal") != {"AWS": "*"}


def test_worker_reserved_concurrency_and_vpc(templates):
    core = templates["core"]
    core.has_resource_properties("AWS::Lambda::Function", {
        "FunctionName": "amx-gpu-bench-worker", "ReservedConcurrentExecutions": 80, "MemorySize": 1769,
        "Architectures": ["arm64"], "VpcConfig": Match.object_like({"SecurityGroupIds": Match.any_value()})})


def test_ec2_start_stop_are_tag_conditioned(templates):
    found = 0
    for _, _, pol in _all(templates, "AWS::IAM::Policy"):
        for st in pol["Properties"]["PolicyDocument"]["Statement"]:
            acts = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
            if any(a in ("ec2:StartInstances", "ec2:StopInstances", "ec2:TerminateInstances") for a in acts):
                found += 1
                assert st["Condition"]["StringEquals"]["aws:ResourceTag/bench:managed"] == "true"
            if "ssm:SendCommand" in acts and "instance" in json.dumps(st["Resource"]):
                assert st["Condition"]["StringEquals"]["ssm:resourceTag/bench:managed"] == "true"
            assert "*" not in acts and not any(a.endswith(":*") for a in acts)
    assert found >= 3


def test_every_taggable_resource_has_project_tags(templates):
    untagged = []
    for name, t in templates.items():
        for lid, r in t.to_json()["Resources"].items():
            tags = r.get("Properties", {}).get("Tags")
            if tags is None:
                continue
            keys = {x["Key"] for x in tags} if isinstance(tags, list) else set(tags)
            if not set(PROJECT_TAGS) <= keys:
                untagged.append(f"{name}/{lid}")
    assert untagged == []


def test_instance_role_scopes(templates):
    t = templates["t2-m8i-4xl"].to_json()["Resources"]
    stmts = [st for r in t.values() if r["Type"] == "AWS::IAM::Policy"
             for st in r["Properties"]["PolicyDocument"]["Statement"]]
    text = json.dumps(stmts)
    assert "secretsmanager:GetSecretValue" in text and "hf-token-naqygX" in text
    assert "cloudwatch:namespace" in text
    puts = [st for st in stmts if st["Action"] == "s3:PutObject"]
    assert puts and all("runs/*/raw" not in json.dumps(st["Resource"]) for st in puts)


# ------------------------------------------------------------------ AmxBench-WebApp (2026-11-15)

def _web(templates):
    return templates["webapp"].to_json()["Resources"]


def test_webapp_user_pool_admits_nobody_on_its_own(templates):
    """No self sign-up, email sign-in, strong passwords, MFA offered, deletion-protected."""
    pool = next(r for r in _web(templates).values() if r["Type"] == "AWS::Cognito::UserPool")["Properties"]
    assert pool["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is True
    assert pool["UsernameAttributes"] == ["email"] and pool["AutoVerifiedAttributes"] == ["email"]
    pw = pool["Policies"]["PasswordPolicy"]
    assert pw["MinimumLength"] >= 8 and all(pw[k] for k in ("RequireLowercase", "RequireUppercase", "RequireNumbers", "RequireSymbols"))
    assert pool["MfaConfiguration"] == "OPTIONAL" and pool["DeletionProtection"] == "ACTIVE"
    assert pool["AccountRecoverySetting"]["RecoveryMechanisms"] == [{"Name": "verified_email", "Priority": 1}]
    groups = [r for r in _web(templates).values() if r["Type"] == "AWS::Cognito::UserPoolGroup"]
    assert [g["Properties"]["GroupName"] for g in groups] == ["operators"]


def test_webapp_client_is_public_code_grant_only_bound_to_the_service_url(templates):
    cl = next(r for r in _web(templates).values() if r["Type"] == "AWS::Cognito::UserPoolClient")["Properties"]
    assert cl["GenerateSecret"] is False                                   # PKCE in the browser, no secret to leak
    assert cl["AllowedOAuthFlows"] == ["code"] and cl["AllowedOAuthFlowsUserPoolClient"] is True
    assert set(cl["AllowedOAuthScopes"]) == {"openid", "email", "profile"}
    assert cl["ExplicitAuthFlows"] == ["ALLOW_REFRESH_TOKEN_AUTH"]          # no password or admin flows
    assert cl["PreventUserExistenceErrors"] == "ENABLED" and cl["EnableTokenRevocation"] is True
    assert cl["SupportedIdentityProviders"] == ["COGNITO"]
    cb = json.dumps(cl["CallbackURLs"])
    assert "DomainName" in cb and cb.count('"/callback"') == 1 and "localhost" not in cb and "http://" not in cb
    units = cl["TokenValidityUnits"]
    mins = {"minutes": 1, "hours": 60, "days": 1440}
    assert cl["AccessTokenValidity"] * mins[units["AccessToken"]] == 60           # one hour
    assert cl["RefreshTokenValidity"] * mins[units["RefreshToken"]] == 12 * 60    # half a day, then sign in again


def test_webapp_is_private_behind_cloudfront_and_gated_by_the_app(templates):
    """The task has no public address and takes its port only from the load balancer; the load
    balancer is internal and takes port 80 only from CloudFront's origin-facing addresses;
    CloudFront is the one public entry (HTTPS, WAF). The container gets the pool, never a credential."""
    res = _web(templates)
    svc = next(r for r in res.values() if r["Type"] == "AWS::ECS::Service")["Properties"]
    assert svc["NetworkConfiguration"]["AwsvpcConfiguration"]["AssignPublicIp"] == "DISABLED"
    assert svc["DesiredCount"] == 1 and svc["DeploymentConfiguration"]["DeploymentCircuitBreaker"] == {"Enable": True, "Rollback": True}
    lb = next(r for r in res.values() if r["Type"] == "AWS::ElasticLoadBalancingV2::LoadBalancer")["Properties"]
    assert lb["Scheme"] == "internal" and lb["Type"] == "network" and lb["SecurityGroups"]
    ingress = [r["Properties"] for r in res.values() if r["Type"] == "AWS::EC2::SecurityGroupIngress"]
    assert sorted((i["FromPort"], "prefix" if "SourcePrefixListId" in i else "sg") for i in ingress) == [(80, "prefix"), (8080, "sg")]
    for r in res.values():
        if r["Type"] == "AWS::EC2::SecurityGroup":
            assert not r["Properties"].get("SecurityGroupIngress")                   # no CIDR ingress anywhere
    dist = next(r for r in res.values() if r["Type"] == "AWS::CloudFront::Distribution")["Properties"]["DistributionConfig"]
    assert dist["Origins"][0]["VpcOriginConfig"] and len(dist["Origins"]) == 1
    assert dist["DefaultCacheBehavior"]["ViewerProtocolPolicy"] == "redirect-to-https" and "WebACLId" in dist
    assert dist["DefaultCacheBehavior"]["AllowedMethods"] == ["GET", "HEAD", "OPTIONS", "PUT", "PATCH", "POST", "DELETE"]
    acl = next(r for r in res.values() if r["Type"] == "AWS::WAFv2::WebACL")["Properties"]
    assert acl["Scope"] == "CLOUDFRONT"
    assert {"RateLimitPerIp", "AWSManagedRulesAmazonIpReputationList", "AWSManagedRulesCommonRuleSet"} <= {r["Name"] for r in acl["Rules"]}
    td = next(r for r in res.values() if r["Type"] == "AWS::ECS::TaskDefinition")["Properties"]
    c = td["ContainerDefinitions"][0]
    env = {e["Name"]: e["Value"] for e in c["Environment"]}
    assert {"AMXBENCH_COGNITO_USER_POOL_ID", "AMXBENCH_COGNITO_DOMAIN", "AMXBENCH_COGNITO_CLIENT_ID", "AMXBENCH_OUTPUTS",
            "AMXBENCH_BIND"} <= set(env)
    assert env["AMXBENCH_CREDS_FILE"] == "" and not any(k.startswith("AWS_") for k in env) and "Secrets" not in c
    assert "/api/health" in " ".join(c["HealthCheck"]["Command"]) and td["RequiresCompatibilities"] == ["FARGATE"]
    tg = next(r for r in res.values() if r["Type"] == "AWS::ElasticLoadBalancingV2::TargetGroup")["Properties"]
    assert tg["HealthCheckPath"] == "/api/health" and tg["HealthCheckProtocol"] == "HTTP"


def test_webapp_role_is_least_privilege(templates):
    """Every statement names its resources, except the describe/list actions that have none; the
    only destructive EC2 action is conditioned on the bench tag; no admin, IAM or wildcard action."""
    res = _web(templates)
    pol = [r for r in res.values() if r["Type"] == "AWS::IAM::Policy"]
    stmts = [s for p in pol for s in p["Properties"]["PolicyDocument"]["Statement"]]
    actions = {a for s in stmts for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])}
    assert not any(a.endswith("*") or a.startswith("iam:") or a.startswith("sts:AssumeRole") for a in actions), actions
    for s in stmts:
        acts = [s["Action"]] if isinstance(s["Action"], str) else s["Action"]
        if s["Resource"] == "*":
            assert all(a.split(":")[1].startswith(("Describe", "List", "GetAuthorizationToken")) for a in acts), acts
        if "ec2:TerminateInstances" in acts:
            assert s["Condition"] == {"StringEquals": {"aws:ResourceTag/bench:managed": "true"}}
        if "s3:ListBucket" in acts:
            assert set(s["Condition"]["StringLike"]["s3:prefix"]) == {"runs/*", "artifacts/prompts/*", "datasets/*",
                                                                      "artifacts/dspm/*", "config/*"}
        if "s3:PutObject" in acts:
            text = json.dumps(s["Resource"])
            assert text.count("runs/*/run.yaml") == 1 or (        # Setup: the catalog object and a new model's pools
                text.count("config/models.yaml") == 1 and "artifacts/prompts/*" in text and "artifacts/dspm/v1/requests/*" in text)
        if any(a.startswith("secretsmanager:") for a in acts):  # the one secret, by complete ARN; no create/delete/list
            assert set(acts) == {"secretsmanager:PutSecretValue", "secretsmanager:DescribeSecret", "secretsmanager:GetSecretValue"}
            assert "hf-token-naqygX" in json.dumps(s["Resource"])
        if "bedrock:InvokeModel" in acts:                       # Claude and OpenAI only (settings.ALLOWED_PROVIDERS)
            text = json.dumps(s["Resource"])
            assert "anthropic.claude-*" in text and "openai.*" in text and "foundation-model/*" not in text
        if "bedrock-agentcore:InvokeWebSearch" in acts:
            assert s["Resource"].endswith(":aws:tool/web-search.v1") and len(acts) == 1
    assert "ec2:RunInstances" not in actions and "ec2:StopInstances" not in actions
    assert "cloudformation:DescribeStacks" not in actions                  # outputs come from the environment
    assert {"dynamodb:Scan", "states:StartExecution", "states:StopExecution", "lambda:GetFunctionConfiguration",
            "bedrock:InvokeModel", "bedrock-agentcore:InvokeWebSearch", "secretsmanager:PutSecretValue"} <= actions
    env = {e["Name"]: e["Value"] for r in res.values() if r["Type"] == "AWS::ECS::TaskDefinition"
           for e in r["Properties"]["ContainerDefinitions"][0]["Environment"]}
    assert "hf-token-naqygX" in env["AMXBENCH_HF_SECRET_ARN"]
    task_roles = [r for r in res.values() if r["Type"] == "AWS::IAM::Role"
                  and r["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]["Service"] == "ecs-tasks.amazonaws.com"]
    assert len(task_roles) == 2 and all("ManagedPolicyArns" not in r["Properties"] for r in task_roles)   # task + execution
