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
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
DRAWIO_APP = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")

AWS = {"compute": "#ED7100", "storage": "#7AA116", "database": "#3B48CC", "network": "#8C4FFF", "security": "#DD344C",
       "management": "#E7157B", "integration": "#E7157B", "ml": "#01A88D"}
GROUP = ("outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=18;fontStyle=1;container=1;pointerEvents=0;"
         "collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;verticalAlign=top;align=left;spacingLeft=30;")
TEXT = "text;html=1;whiteSpace=wrap;align=center;verticalAlign=top;fontSize=15;fontColor=#232F3E;"
BOX = "rounded=1;whiteSpace=wrap;html=1;fontSize=15;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;fontSize=14;fontColor=#232F3E;strokeColor=#232F3E;strokeWidth=2;labelBackgroundColor=#FFFFFF;"


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
                 f"fontStyle=1;fontSize=18;spacingLeft=8;spacingTop=4;fontColor={stroke};container=1;")
        self.cells.append(f'<mxCell id="{cid}" value="{_label(label)}" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def icon(self, cid: str, name: str, color: str, label: str, x: int, y: int, *, parent: str = "1", label_w: int = 190) -> str:
        style = (f"sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor={color};strokeColor=none;dashed=0;"
                 f"verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=15;aspect=fixed;"
                 f"shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{name};")
        self.cells.append(f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x}" y="{y}" width="64" height="64" as="geometry"/></mxCell>')
        tid = self._id("t")
        self.cells.append(f'<mxCell id="{tid}" value="{_label(label)}" style="{TEXT}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x + 32 - label_w // 2}" y="{y + 68}" width="{label_w}" height="80" as="geometry"/></mxCell>')
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
    """Labels are a name and one short line: at page width a 15 px label renders at about 11 px, and the Technical
    details page carries the long explanations. Every cell is a child of the root layer so edges route freely."""
    d = Diagram()
    # ---- containers
    d.group("cloud", "AWS account · us-east-1 (CDK stacks AmxBench-Network, -Core, -WebApp)", 300, 20, 1560, 1120,
            icon="group_aws_cloud_alt", stroke="#232F3E")
    d.plain_group("entry", "Public entry", 330, 70, 400, 260, "#8C4FFF")
    d.plain_group("orch", "Run orchestration (Core stack)", 760, 70, 640, 260, "#E7157B")
    d.plain_group("data", "State, data and images", 1430, 70, 410, 640, "#7AA116")
    d.plain_group("ai", "Reasoning", 1430, 750, 410, 350, "#01A88D")
    d.group("vpc", "Bench VPC: private subnets, NAT egress, no public IPs", 330, 370, 1070, 730,
            icon="group_vpc2", stroke="#8C4FFF", font="#8C4FFF")
    d.group("subnet", "Private subnet", 355, 420, 1020, 655, icon="group_security_group", stroke="#00A4A6", fill="#E6F6F7", font="#147EBA")
    d.plain_group("web", "Web application (WebApp stack)", 380, 470, 350, 260, "#147EBA")
    d.group("hosts", "Test hosts: one EC2 instance per hardware option, per run", 760, 470, 590, 580,
            icon="group_spot_fleet", stroke="#D86613")
    # ---- outside AWS
    d.box("user", "Operator's browser\nsigned in through Cognito", 40, 130, 220, 90, "fillColor=#F5F5F5;strokeColor=#666666;")
    d.box("hf", "Hugging Face Hub\npinned model revisions", 40, 900, 220, 80, "fillColor=#FFF2CC;strokeColor=#D6B656;")
    # ---- public entry
    d.icon("cf", "cloudfront", AWS["network"], "CloudFront + WAF\nHTTPS, rate limits", 370, 120)
    d.icon("cognito", "cognito", AWS["security"], "Cognito user pool\nhosted login, no self sign-up", 570, 120)
    # ---- orchestration
    d.icon("sfn", "step_functions", AWS["integration"], "Step Functions\nrun and level state machines", 800, 120)
    d.icon("lambda", "lambda", AWS["compute"], "Lambdas\nplan, load workers, probes, aggregate", 1010, 120)
    d.icon("ssm", "systems_manager", AWS["management"], "Systems Manager\nRun Command to the hosts", 1220, 120)
    # ---- data
    d.icon("s3", "s3", AWS["storage"], "S3 bucket\nruns, prompt pools, model cache, catalog", 1470, 120)
    d.icon("ddb", "dynamodb", AWS["database"], "DynamoDB\nrun, cell and level status", 1680, 120)
    d.icon("secrets", "secrets_manager", AWS["security"], "Secrets Manager\nHugging Face token", 1470, 320)
    d.icon("cw", "cloudwatch", AWS["management"], "CloudWatch\nhost metrics every 10 s, logs", 1680, 320)
    d.icon("ecr", "ecr", AWS["compute"], "ECR\nengine and web app images", 1470, 520)
    d.box("pub", "Public registries\nvLLM GPU and CPU images, pinned by digest", 1640, 540, 180, 100,
          "fillColor=#FFFFFF;strokeColor=#7AA116;")
    # ---- reasoning
    d.icon("bedrock", "bedrock", AWS["ml"], "Amazon Bedrock\nClaude: hardware proposal, executive summary\nAgentCore web search", 1570, 810, label_w=260)
    # ---- web app
    d.icon("nlb", "network_load_balancer", AWS["network"], "Internal NLB\nreached only from CloudFront", 400, 530, label_w=150)
    d.icon("fargate", "fargate", AWS["compute"], "ECS Fargate task\nFastAPI API + React UI", 580, 530, label_w=150)
    # ---- hosts
    d.icon("gpu", "ec2", AWS["compute"], "GPU host\ng6 (L4) or g6e (L40S); vLLM", 800, 530, label_w=180)
    d.icon("cpu", "ec2", AWS["compute"], "AMX CPU host\nm8i / c8i Xeon 6; vLLM CPU", 1020, 530, label_w=180)
    d.icon("agent", "cloudwatch", AWS["management"], "CloudWatch agent\nCPU, memory, GPU samples", 1230, 530, label_w=150)
    d.box("flow", "On every host, per model and number format:\n"
                  "1. stage the pinned weights (S3 cache, else Hugging Face)\n"
                  "2. start the engine container, check its health\n"
                  "3. Lambdas send the load over HTTP inside the VPC\n"
                  "4. perf counters prove AMX use; raw records go to S3",
          790, 740, 530, 270, "align=left;verticalAlign=top;spacing=12;fillColor=#FFFFFF;strokeColor=#9673A6;")
    # ---- flows
    d.edge("user", "cf", "HTTPS")
    d.edge("cf", "cognito", "sign-in", dashed=True)
    d.edge("cf", "nlb", "VPC origin", exit=(0.5, 1), entry=(0.5, 0))
    d.edge("nlb", "fargate", "HTTP 8080")
    d.edge("web", "orch", "start / stop runs", exit=(0.75, 0), entry=(0.25, 1), via=[(642, 350), (920, 350)])
    d.edge("web", "data", "state, data, metrics", exit=(0.3, 0), entry=(0, 0.6), via=[(485, 395), (1410, 395), (1410, 454)])
    d.edge("web", "ai", "proposal, summary", exit=(0.5, 1), entry=(0, 0.5), via=[(555, 1085), (1410, 1085), (1410, 925)])
    d.edge("orch", "hosts", "launch, commands, load", exit=(0.75, 1), entry=(0.5, 0))
    d.edge("orch", "data", "plan, metrics, state", exit=(1, 0.3), entry=(0, 0.08))
    d.edge("hosts", "data", "model cache, records, images", exit=(1, 0.5), entry=(0, 0.85))
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
        subprocess.run([str(app), "-x", "-f", fmt, "-b", "16", *extra, "-o", str(out), str(drawio)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if fmt == "svg":
            out.write_text(fixed_light(out.read_text()))
        print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


def fixed_light(svg: str) -> str:
    """The export declares `color-scheme: light dark` and a light-dark() foreground, so a browser in dark mode drew the
    text light grey on the light fills and the groups' dark fills (unreadable, 2026-11-22). Pin the picture to its
    light colours and give it a white background; the page shows it in a white frame either way."""
    svg = re.sub(r"color-scheme:\s*light dark;?", "", svg)
    svg = re.sub(r"<style type=\"text/css\">@supports \(color: light-dark\(#000, #fff\)\)[^<]*</style>", "", svg)
    svg = re.sub(r"var\(--ge-adaptive-fg(?:, *(#[0-9a-fA-F]{3,6}))?\)", lambda m: m.group(1) or "#000000", svg)
    # keep the light colour of every pair; a colour may itself be a function call (rgb(...))
    svg = re.sub(r"light-dark\(((?:[^,()]|\([^()]*\))+), *(?:[^()]|\([^()]*\))+\)", r"\1", svg)
    return svg.replace("background: transparent; background-color: transparent;", "background: #FFFFFF; background-color: #FFFFFF;")


if __name__ == "__main__":
    sys.exit(main())
