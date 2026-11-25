"""The tags every resource of this project carries.
`cdk.Tags.of(app)` puts them on every taggable CloudFormation resource of every stack; lambdas/target_control.py
puts them on the EC2 instances and volumes it launches per run (RunInstances TagSpecifications), which no stack
owns. `COST_ALLOCATION_TAG` is the key activated as an AWS cost allocation tag (infra/stacks/core.py), so every
dollar this project spends can be grouped by it in Cost Explorer and the Cost and Usage Report.
An `owner` tag is added only when AMXBENCH_OWNER is set (at `cdk deploy` time for the stacks' resources, in the
Lambda environment for launched instances), so a public checkout names nobody by default.
"""
from __future__ import annotations

import os

COST_ALLOCATION_TAG = "Amx"
PROJECT_TAGS: dict[str, str] = {
    COST_ALLOCATION_TAG: "amx-gpu-bench",
    "project": "amx-gpu-bench",
    "purpose": "cpu-amx-vs-gpu-benchmark",
}
if os.environ.get("AMXBENCH_OWNER", "").strip():
    PROJECT_TAGS["owner"] = os.environ["AMXBENCH_OWNER"].strip()
