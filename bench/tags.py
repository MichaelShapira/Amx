"""The tags every resource of this project carries.

`cdk.Tags.of(app)` puts them on every taggable CloudFormation resource of every stack; lambdas/target_control.py
puts them on the EC2 instances and volumes it launches per run (RunInstances TagSpecifications), which no stack
owns. `COST_ALLOCATION_TAG` is the key activated as an AWS cost allocation tag (infra/stacks/core.py), so every
dollar this project spends can be grouped by it in Cost Explorer and the Cost and Usage Report.
"""
from __future__ import annotations

COST_ALLOCATION_TAG = "Amx"
PROJECT_TAGS: dict[str, str] = {
    COST_ALLOCATION_TAG: "amx-gpu-bench",
    "project": "amx-gpu-bench",
    "owner": "operator",
    "purpose": "cpu-amx-vs-gpu-benchmark",
}
