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
