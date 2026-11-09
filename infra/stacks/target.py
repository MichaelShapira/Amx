"""TargetStack, one per target (design 6, 8): an EC2 instance in the private subnet with
IMDSv2, no public IP, no key pair and an encrypted gp3 root volume. User data installs
Docker, perf and the CloudWatch agent, syncs /opt/bench from S3 and records host facts on
every boot. The instance ID and AMI ID go to SSM for preflight and the control Lambdas."""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack, Tags
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from bench.config import TargetSpec
from infra.stacks.core import CoreStack
from infra.stacks.network import NetworkStack


def user_data(target: TargetSpec, bucket: str, region: str) -> str:
    gpu = target.accelerator == "gpu"
    drop_gpu = "" if gpu else " | jq 'del(.metrics.metrics_collected.nvidia_gpu)'"
    return f"""#!/bin/bash
set -euo pipefail
export AWS_DEFAULT_REGION={region}
dnf install -y docker jq perf gcc numactl amazon-cloudwatch-agent || dnf install -y jq perf gcc numactl
systemctl enable --now docker
install -d -m 0755 /opt/bench /opt/models /var/lib/bench
printf 'export AWS_DEFAULT_REGION=%s\\nexport BENCH_BUCKET=%s\\n' {region} {bucket} >/opt/bench/env
cat >/usr/local/bin/bench-boot <<'EOF'
#!/bin/bash
set -euo pipefail
source /opt/bench/env
aws s3 sync --only-show-errors "s3://$BENCH_BUCKET/artifacts/host/" /opt/bench/ --exclude env
chmod +x /opt/bench/*.sh
sed 's/__TARGET_ID__/{target.id}/g' /opt/bench/cloudwatch-agent.json{drop_gpu} >/opt/bench/cwagent.json
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c file:/opt/bench/cwagent.json || true
/opt/bench/host_facts.sh {target.id} "$BENCH_BUCKET"
EOF
chmod 0755 /usr/local/bin/bench-boot
cat >/etc/systemd/system/bench-boot.service <<'EOF'
[Unit]
Description=Benchmark host setup and host facts
After=network-online.target docker.service
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/bench-boot
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now bench-boot.service
"""


class TargetStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, target: TargetSpec, network: NetworkStack, core: CoreStack,
                 instance_type: str | None = None, **kw):
        super().__init__(scope, cid, **kw)
        itype = instance_type or target.instance_type
        role = iam.Role(self, "InstanceRole", assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
                        managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")])
        core.instance_policy(role, target.id)
        image = ec2.MachineImage.from_ssm_parameter(target.ami_ssm, os=ec2.OperatingSystemType.LINUX)
        ud = ec2.UserData.custom(user_data(target, core.bucket.bucket_name, self.region))
        inst = ec2.Instance(
            self, "Instance", instance_type=ec2.InstanceType(itype), machine_image=image, vpc=network.vpc,
            vpc_subnets=network.private_subnets, security_group=network.sg_targets, role=role, user_data=ud,
            require_imdsv2=True, associate_public_ip_address=False, ssm_session_permissions=False,
            block_devices=[ec2.BlockDevice(device_name="/dev/xvda", volume=ec2.BlockDeviceVolume.ebs(
                target.root_volume_gib, encrypted=True, volume_type=ec2.EbsDeviceVolumeType.GP3,
                delete_on_termination=True))],
            detailed_monitoring=False)
        Tags.of(inst).add("bench:managed", "true")
        Tags.of(inst).add("bench:target", target.id)
        Tags.of(inst).add("Name", f"amx-gpu-bench-{target.id}")
        ssm.StringParameter(self, "InstanceIdParam", parameter_name=f"/amx-gpu-bench/targets/{target.id}",
                            string_value=inst.instance_id)
        ami_id = image.get_image(self).image_id
        ssm.StringParameter(self, "AmiParam", parameter_name=f"/amx-gpu-bench/targets/{target.id}/ami",
                            string_value=ami_id)
        CfnOutput(self, "InstanceId", value=inst.instance_id)
        CfnOutput(self, "AmiId", value=ami_id)
        CfnOutput(self, "InstanceType", value=itype)
        self.instance = inst
