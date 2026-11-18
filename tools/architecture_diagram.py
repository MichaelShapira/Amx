"""Write docs/architecture.drawio, the application's architecture as a draw.io diagram, and
export it to SVG and PNG with draw.io desktop when it is installed.

    .venv/bin/python tools/architecture_diagram.py            # writes docs/architecture.{drawio,svg,png}

The diagram is data in this file, so a change to the architecture is a change here, reviewed
like code; the web app's Technical details page serves docs/architecture.svg (webui/api/main.py).
Icons are draw.io's AWS 2024 resource icons (mxgraph.aws4.*); the export needs
/Applications/draw.io.app (brew install --cask drawio). Without it only the .drawio is written.
"""

from __future__ import annotations

import html
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
DRAWIO_APP = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")

AWS = {"compute": "#ED7100", "storage": "#7AA116", "database": "#3B48CC", "network": "#8C4FFF", "security": "#DD344C",
       "management": "#E7157B", "integration": "#E7157B", "ml": "#01A88D"}
GROUP = ("outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;"
         "collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;verticalAlign=top;align=left;spacingLeft=30;")
TEXT = "text;html=1;whiteSpace=wrap;align=center;verticalAlign=top;fontSize=10;fontColor=#232F3E;"
BOX = "rounded=1;whiteSpace=wrap;html=1;fontSize=11;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=10;fontColor=#232F3E;strokeColor=#232F3E;labelBackgroundColor=#FFFFFF;"


def _label(text: str) -> str:
    """Escape for an XML attribute; newlines become <br> (every cell here renders HTML)."""
    return html.escape(text).replace("\n", "&lt;br&gt;")


class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []
        self.n = 1

    def _id(self, prefix: str) -> str:
        self.n += 1
        return f"{prefix}{self.n}"

    def group(self, cid: str, label: str, x: int, y: int, w: int, h: int, *, icon: str, stroke: str, fill: str = "none",
              font: str | None = None, parent: str = "1") -> str:
        style = GROUP + f"grIcon=mxgraph.aws4.{icon};strokeColor={stroke};fillColor={fill};fontColor={font or stroke};"
        self.cells.append(f'<mxCell id="{cid}" value="{_label(label)}" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def plain_group(self, cid: str, label: str, x: int, y: int, w: int, h: int, stroke: str, parent: str = "1") -> str:
        style = (f"rounded=1;whiteSpace=wrap;html=1;dashed=1;strokeColor={stroke};fillColor=none;verticalAlign=top;align=left;"
                 f"fontStyle=1;fontSize=12;spacingLeft=8;spacingTop=2;fontColor={stroke};container=1;")
        self.cells.append(f'<mxCell id="{cid}" value="{_label(label)}" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def icon(self, cid: str, name: str, color: str, label: str, x: int, y: int, *, parent: str = "1", label_w: int = 150) -> str:
        style = (f"sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor={color};strokeColor=none;dashed=0;"
                 f"verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=10;aspect=fixed;"
                 f"shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{name};")
        self.cells.append(f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="52" height="52" as="geometry"/></mxCell>')
        tid = self._id("t")
        self.cells.append(f'<mxCell id="{tid}" value="{_label(label)}" style="{TEXT}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x + 26 - label_w // 2}" y="{y + 56}" width="{label_w}" height="60" as="geometry"/></mxCell>')
        return cid

    def box(self, cid: str, label: str, x: int, y: int, w: int, h: int, style: str = "", parent: str = "1") -> str:
        self.cells.append(f'<mxCell id="{cid}" value="{_label(label)}" style="{BOX}{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def edge(self, src: str, dst: str, label: str = "", *, dashed: bool = False, exit: tuple[float, float] | None = None,
             entry: tuple[float, float] | None = None, via: list[tuple[int, int]] | None = None, parent: str = "1") -> None:
        style = EDGE + ("dashed=1;" if dashed else "")
        if exit:
            style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
        if entry:
            style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in (via or []))
        geo = f'<mxGeometry relative="1" as="geometry"><Array as="points">{pts}</Array></mxGeometry>' if pts else '<mxGeometry relative="1" as="geometry"/>'
        self.cells.append(f'<mxCell id="{self._id("e")}" value="{_label(label)}" style="{style}" edge="1" parent="{parent}" '
                          f'source="{src}" target="{dst}">{geo}</mxCell>')

    def xml(self, name: str) -> str:
        body = "".join(self.cells)
        return (f'<mxfile host="tools/architecture_diagram.py"><diagram name="{html.escape(name)}" id="arch">'
                f'<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
                f'page="1" pageScale="1" pageWidth="1600" pageHeight="1000" math="0" shadow="0"><root>'
                f'<mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')


def build() -> Diagram:
    d = Diagram()
    # ---- containers (absolute coordinates: every cell is a child of the root layer, so edges route freely)
    d.group("cloud", "AWS account · us-east-1 (CDK stacks AmxBench-Network, -Core, -WebApp)", 240, 20, 1270, 940,
            icon="group_aws_cloud_alt", stroke="#232F3E")
    d.plain_group("entry", "Public entry", 270, 60, 300, 230, "#8C4FFF")
    d.plain_group("orch", "Run orchestration (Core stack)", 600, 60, 520, 230, "#E7157B")
    d.plain_group("data", "State, data and images", 1150, 60, 340, 540, "#7AA116")
    d.plain_group("ai", "Reasoning", 1150, 630, 340, 300, "#01A88D")
    d.group("vpc", "Bench VPC: private subnets with NAT egress; nothing has a public IP", 270, 320, 850, 610,
            icon="group_vpc2", stroke="#8C4FFF", font="#8C4FFF")
    d.group("subnet", "Private subnet", 290, 360, 810, 550, icon="group_security_group", stroke="#00A4A6", fill="#E6F6F7", font="#147EBA")
    d.plain_group("web", "Web application (WebApp stack)", 310, 400, 270, 220, "#147EBA")
    d.group("hosts", "Test hosts: one EC2 instance per hardware option, per run (Spot, On-Demand fallback)",
            600, 400, 480, 480, icon="group_spot_fleet", stroke="#D86613")

    # ---- outside AWS
    d.box("user", "Operator's browser\n(any network; identity from the Cognito user pool, operators group may start runs)", 40, 120, 170, 90,
          "fillColor=#F5F5F5;strokeColor=#666666;")
    d.box("hf", "Hugging Face Hub\npinned model revisions; token only for gated models", 40, 760, 170, 80, "fillColor=#FFF2CC;strokeColor=#D6B656;")

    # ---- public entry
    d.icon("cf", "cloudfront", AWS["network"], "CloudFront + WAF\nHTTPS only; rate limit, IP reputation, common rules", 300, 100)
    d.icon("cognito", "cognito", AWS["security"], "Cognito user pool\nhosted login, code + PKCE; no self sign-up", 460, 100)

    # ---- orchestration
    d.icon("sfn", "step_functions", AWS["integration"], "Step Functions\nrun state machine; one level state machine per measurement", 640, 100)
    d.icon("lambda", "lambda", AWS["compute"], "Lambdas\npreflight · target_control · init_level · plan_wave · worker ×N (load) · warmup · quality · perf_counters · amx_probe · aggregate_level/run · reaper", 800, 100, label_w=190)
    d.icon("ssm", "systems_manager", AWS["management"], "Systems Manager\nRun Command carries the host scripts' commands; parameters hold target launch templates", 990, 100)

    # ---- data
    d.icon("s3", "s3", AWS["storage"], "S3 bench bucket\nruns/<id>/ (plan, raw records, metrics, report) · artifacts/ (prompt pools, host scripts) · models/ (HF + OpenVINO IR cache) · config/models.yaml (live catalog) · datasets/", 1180, 100, label_w=160)
    d.icon("ddb", "dynamodb", AWS["database"], "DynamoDB run table\nRUN# · CELL# · LEVEL# rows; SETTINGS (reasoning model, token status)", 1350, 100)
    d.icon("secrets", "secrets_manager", AWS["security"], "Secrets Manager\nHugging Face token (written from Setup; read by hosts and the onboarding job)", 1180, 280)
    d.icon("cw", "cloudwatch", AWS["management"], "CloudWatch\nAmxGpuBench metrics (CPU, memory, GPU per host, 10 s); Lambda and app logs", 1350, 280)
    d.icon("ecr", "ecr", AWS["compute"], "ECR (CDK assets)\nthin-torch-cpu · thin-torch-cuda · thin-openvino · export · web app images", 1180, 460)
    d.box("pub", "Public registries (pinned by digest)\nvLLM GPU (ECR Public DLC) · vLLM CPU (Docker Hub) · OpenVINO Model Server (Docker Hub)", 1300, 470, 170, 90,
          "fillColor=#FFFFFF;strokeColor=#7AA116;fontSize=10;")

    # ---- reasoning
    d.icon("bedrock", "bedrock", AWS["ml"], "Amazon Bedrock\nClaude Opus (default; Setup → Reasoning model): hardware proposal, executive summary. AgentCore web search (InvokeWebSearch) grounds the proposal", 1290, 680, label_w=200)

    # ---- web app
    d.icon("nlb", "network_load_balancer", AWS["network"], "Internal NLB\nport 80 from CloudFront's origin-facing addresses only", 335, 450, label_w=120)
    d.icon("fargate", "fargate", AWS["compute"], "ECS Fargate task (×1, task role)\nFastAPI API + React UI: wizard, Setup (catalog, token, model onboarding), Results, utilization, AI jobs", 485, 450, label_w=150)

    # ---- hosts
    d.icon("gpu", "ec2", AWS["compute"], "GPU host (g6 L4, g6e L40S)\nvLLM GPU image or thin-torch-cuda", 640, 450)
    d.icon("cpu", "ec2", AWS["compute"], "AMX CPU host (m8i, c8i: Xeon 6)\nvLLM CPU · OpenVINO Model Server · thin-openvino · thin-torch-cpu", 820, 450)
    d.icon("agent", "cloudwatch", AWS["management"], "CloudWatch agent\non every host: cpu-total, memory, nvidia_smi every 10 s", 1000, 450)
    d.box("flow", "On every host, per model × number format (scripts synced from s3://…/artifacts/host/ at boot):\n"
                  "1. stage_model.sh — pinned weights from the S3 cache, else from Hugging Face (sha256-checked), then cached\n"
                  "2. OpenVINO formats only: export container (optimum-intel / OVMS export_model.py) → IR, cached under models/ov/<hash>\n"
                  "3. serve.sh — docker run of the engine image with the cell's options; health check; one real request\n"
                  "4. Lambdas drive the load over HTTP inside the VPC; perf counters prove AMX use; raw records → S3",
          620, 610, 440, 250, "align=left;verticalAlign=top;spacing=8;fontSize=10;fillColor=#FFFFFF;strokeColor=#9673A6;")

    # ---- flows (short labels; the Technical details page explains each)
    d.edge("user", "cf", "HTTPS")
    d.edge("cf", "cognito", "sign-in", dashed=True)
    d.edge("cf", "nlb", "VPC origin", exit=(0.5, 1), entry=(0.5, 0))
    d.edge("nlb", "fargate", "HTTP 8080")
    d.edge("web", "orch", "start / stop runs", exit=(0.75, 0), entry=(0.25, 1), via=[(512, 300), (730, 300)])
    d.edge("web", "data", "state, data, secret, metrics", exit=(0.3, 0), entry=(0, 0.6), via=[(391, 340), (1125, 340), (1125, 384)])
    d.edge("web", "ai", "proposal, summary", exit=(0.5, 1), entry=(0, 0.5), via=[(445, 900), (1125, 900), (1125, 780)])
    d.edge("orch", "hosts", "launch, SSM commands, HTTP load", exit=(0.75, 1), entry=(0.5, 0))
    d.edge("orch", "data", "plan, metrics, state", exit=(1, 0.3), entry=(0, 0.08))
    d.edge("hosts", "data", "model cache, records, metrics, images", exit=(1, 0.5), entry=(0, 0.85))
    d.edge("hosts", "hf", "cache miss", dashed=True, exit=(0, 0.95), entry=(1, 0.5))
    return d


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = build()
    drawio = OUT / "architecture.drawio"
    drawio.write_text(d.xml("AMX GPU bench architecture"))
    print(f"wrote {drawio}")
    app = DRAWIO_APP if DRAWIO_APP.exists() else (Path(shutil.which("drawio")) if shutil.which("drawio") else None)
    if not app:
        print("draw.io desktop not found; .drawio written, no SVG/PNG export (brew install --cask drawio)")
        return 0
    for fmt, extra in (("svg", []), ("png", ["-s", "2"])):
        out = OUT / f"architecture.{fmt}"
        subprocess.run([str(app), "-x", "-f", fmt, "-b", "10", *extra, "-o", str(out), str(drawio)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
