"""NetworkStack (design 6): one private subnet for targets and in-VPC Lambdas, one public
subnet for the NAT gateway, S3 and DynamoDB gateway endpoints, and two security groups.
The only ingress anywhere is TCP 8000 into sg-targets from sg-workers."""

from __future__ import annotations

from aws_cdk import Stack
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

ENGINE_PORT = 8000


class NetworkStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, az_name: str, **kw):
        super().__init__(scope, cid, **kw)
        self.vpc = ec2.Vpc(
            self, "Vpc", ip_addresses=ec2.IpAddresses.cidr("10.40.0.0/16"), availability_zones=[az_name],
            nat_gateways=1, restrict_default_security_group=True,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public-nat", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24,
                                        map_public_ip_on_launch=False),
                ec2.SubnetConfiguration(name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=20),
            ],
            gateway_endpoints={"S3": ec2.GatewayVpcEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.S3),
                               "DynamoDB": ec2.GatewayVpcEndpointOptions(
                                   service=ec2.GatewayVpcEndpointAwsService.DYNAMODB)})
        self.sg_workers = ec2.SecurityGroup(self, "SgWorkers", vpc=self.vpc, allow_all_outbound=False,
                                            description="bench load-generator Lambdas: egress to targets and AWS only")
        self.sg_targets = ec2.SecurityGroup(self, "SgTargets", vpc=self.vpc, allow_all_outbound=False,
                                            description="bench inference targets: port 8000 from workers only")
        self.sg_targets.add_ingress_rule(self.sg_workers, ec2.Port.tcp(ENGINE_PORT), "engine port from workers")
        self.sg_workers.add_egress_rule(self.sg_targets, ec2.Port.tcp(ENGINE_PORT), "engine port on targets")
        # Workers reach S3 and DynamoDB through the gateway endpoints (AWS prefix lists).
        self.sg_workers.add_egress_rule(ec2.Peer.prefix_list(self._prefix_list("s3")), ec2.Port.tcp(443), "S3")
        self.sg_workers.add_egress_rule(ec2.Peer.prefix_list(self._prefix_list("dynamodb")), ec2.Port.tcp(443),
                                        "DynamoDB")
        # Targets download images, models and packages through the NAT gateway; nothing inbound.
        self.sg_targets.add_egress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(443), "HTTPS out via NAT")
        self.private_subnets = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

    def _prefix_list(self, service: str) -> str:
        """AWS-managed prefix list IDs differ per region; resolve them at deploy time."""
        from aws_cdk import custom_resources as cr

        res = cr.AwsCustomResource(
            self, f"PrefixList{service.title()}",
            on_update=cr.AwsSdkCall(service="EC2", action="describePrefixLists",
                                    parameters={"Filters": [{"Name": "prefix-list-name",
                                                             "Values": [f"com.amazonaws.{self.region}.{service}"]}]},
                                    physical_resource_id=cr.PhysicalResourceId.of(f"pl-{service}"),
                                    output_paths=["PrefixLists.0.PrefixListId"]),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE),
            install_latest_aws_sdk=False)
        return res.get_response_field("PrefixLists.0.PrefixListId")
