#!/usr/bin/env python3
"""Renting a GPU on Vast.ai or RunPod for lab.py (from the registry's old validate_rented.py).

The recipe's own digest-pinned image is the container image and its arguments
are the container command, so the launch contract under test is the one the
registry publishes. Rented containers have no docker daemon of their own, so
nothing here is docker-in-docker. Acceptance is the same `accept_recipe.py`
that `local-ai validate` runs locally; the rented box's public endpoint is
the target. The container is always destroyed on exit unless --keep is given.

Standard library only. Credentials: RunPod from ~/.runpod/config.toml or
RUNPOD_API_KEY; Vast from the `vastai` CLI (~/.config/vastai/vast_api_key).

    python3 scripts/validate_rented.py <recipe-id> [--provider vast|runpod]
        [--gpu <provider gpu name>] [--disk 50] [--timeout 3600]
        [--start-timeout 600] [--retries 1] [--keep] [--recommend] [--dry-run]

Provider notes:
  runpod  cannot pull from ghcr.io on community hosts (containers never start);
          fine for Docker Hub images (vllm, sglang). --cloud picks the cloud.
  vast    pulls any public registry through the host's docker; ports are
          mapped to a public ip:port; offers are chosen cheapest-first among
          verified hosts with fast downlink.
"""

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "registry"

# registry hardware id -> provider GPU names. vast names use underscores for spaces in queries.
GPUS = {
    "gtx-1080-ti-11gb": {"vast": "GTX 1080 Ti"},
    "rtx-3060-12gb": {"runpod": ["NVIDIA GeForce RTX 3060"], "vast": "RTX 3060"},
    "rtx-3060-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 3060 Ti"], "vast": "RTX 3060 Ti"},
    "rtx-4060-8gb": {"runpod": ["NVIDIA GeForce RTX 4060"], "vast": "RTX 4060"},
    "rtx-4060-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 4060 Ti"], "vast": "RTX 4060 Ti"},
    "rtx-4060-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 4060 Ti"], "vast": "RTX 4060 Ti"},
    "rtx-5060-8gb": {"runpod": ["NVIDIA GeForce RTX 5060"], "vast": "RTX 5060"},
    "rtx-5060-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 5060 Ti"], "vast": "RTX 5060 Ti"},
    "rtx-3070-8gb": {"runpod": ["NVIDIA GeForce RTX 3070"], "vast": "RTX 3070"},
    "rtx-3070-ti-8gb": {"runpod": ["NVIDIA GeForce RTX 3070 Ti"], "vast": "RTX 3070 Ti"},
    "rtx-3080-10gb": {"runpod": ["NVIDIA GeForce RTX 3080"], "vast": "RTX 3080"},
    "rtx-3080-12gb": {"runpod": ["NVIDIA GeForce RTX 3080"], "vast": "RTX 3080"},
    "rtx-4070-super-12gb": {"runpod": ["NVIDIA GeForce RTX 4070 SUPER"], "vast": "RTX 4070S"},
    "rtx-5060-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 5060 Ti"], "vast": "RTX 5060 Ti"},
    "rtx-3080-ti-12gb": {"runpod": ["NVIDIA GeForce RTX 3080 Ti"], "vast": "RTX 3080 Ti"},
    "rtx-3090-24gb": {"runpod": ["NVIDIA GeForce RTX 3090"], "vast": "RTX 3090"},
    "rtx-3090-ti-24gb": {"runpod": ["NVIDIA GeForce RTX 3090 Ti"], "vast": "RTX 3090 Ti"},
    "rtx-4070-12gb": {"runpod": ["NVIDIA GeForce RTX 4070"], "vast": "RTX 4070"},
    "rtx-4070-ti-12gb": {"runpod": ["NVIDIA GeForce RTX 4070 Ti"], "vast": "RTX 4070 Ti"},
    "rtx-4070-ti-super-16gb": {"runpod": ["NVIDIA GeForce RTX 4070 Ti SUPER"], "vast": "RTX 4070S Ti"},
    "rtx-4080-16gb": {"runpod": ["NVIDIA GeForce RTX 4080"], "vast": "RTX 4080"},
    "rtx-4080-super-16gb": {"runpod": ["NVIDIA GeForce RTX 4080 SUPER"], "vast": "RTX 4080S"},
    "rtx-4090-24gb": {"runpod": ["NVIDIA GeForce RTX 4090"], "vast": "RTX 4090"},
    "rtx-5060-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 5060 Ti"], "vast": "RTX 5060 Ti"},
    "rtx-5070-12gb": {"runpod": ["NVIDIA GeForce RTX 5070"], "vast": "RTX 5070"},
    "rtx-5070-ti-16gb": {"runpod": ["NVIDIA GeForce RTX 5070 Ti"], "vast": "RTX 5070 Ti"},
    "rtx-5080-16gb": {"runpod": ["NVIDIA GeForce RTX 5080"], "vast": "RTX 5080"},
    "rtx-5090-32gb": {"runpod": ["NVIDIA GeForce RTX 5090"], "vast": "RTX 5090"},
    "rtx-2000-ada-16gb": {"runpod": ["NVIDIA RTX 2000 Ada Generation"], "vast": "RTX 2000Ada"},
    "rtx-4000-ada-20gb": {"runpod": ["NVIDIA RTX 4000 Ada Generation"], "vast": "RTX 4000Ada"},
    "rtx-6000-ada-48gb": {"runpod": ["NVIDIA RTX 6000 Ada Generation"], "vast": "RTX 6000Ada"},
    "rtx-a6000-48gb": {"runpod": ["NVIDIA RTX A6000"], "vast": "RTX A6000"},
    "rtx-pro-4000-blackwell-24gb": {"runpod": ["NVIDIA RTX PRO 4000 Blackwell"], "vast": "RTX PRO 4000"},
    "rtx-pro-4500-blackwell-32gb": {"runpod": ["NVIDIA RTX PRO 4500 Blackwell"], "vast": "RTX PRO 4500"},
    "rtx-pro-6000-blackwell-96gb": {"runpod": ["NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition"], "vast": "RTX PRO 6000 WS"},
}


def log(msg):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", file=sys.stderr, flush=True)


def http_probe(url, timeout=10):
    """-> 'ok' | 'refused' (host reachable, server not listening yet) | 'timeout' (port unreachable) | 'http-<code>'."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return "ok" if response.status == 200 else f"http-{response.status}"
    except urllib.error.HTTPError as error:
        return f"http-{error.code}"
    except urllib.error.URLError as error:
        reason = str(error.reason).lower()
        return "timeout" if "timed out" in reason else "refused"
    except (TimeoutError, OSError):
        return "timeout"


class StartStalled(Exception):
    """The host accepted the job but the container never started (image pull stalled or failed)."""


class Spec:
    def __init__(self, recipe, contract, args):
        self.recipe = recipe
        self.contract = contract
        self.image = contract["image"]
        self.entrypoint = contract.get("entrypoint")
        self.arguments = list(contract.get("arguments") or [])
        self.port = int(contract["container_port"])
        self.disk = args.disk
        self.name = f"local-ai-validate-{recipe['id']}"[:191]
        self.env = dict(contract.get("environment") or {})
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if "HF_TOKEN" not in self.env and token_file.exists():
            self.env["HF_TOKEN"] = token_file.read_text(encoding="utf-8").strip()
        hw = json.loads((REG / "hardware" / f"{recipe['hardware_id']}.json").read_text(encoding="utf-8"))
        self.vram_gb = (hw.get("memory") or {}).get("vram_gb") or 0
        self.gpu_override = args.gpu
        self.hardware_id = recipe["hardware_id"]
        self.instance = json.loads((REG / "model-instance" / f"{recipe['model_instance_id']}.json").read_text(encoding="utf-8"))
        self.provision = self.materialize_plan()

    def materialize_plan(self):
        """What the plugin provides through bind mounts, expressed as steps a rented container runs itself.

        `${MODEL_ROOT}/x -> target`: download the pinned weights into target (into the subdirectory the
        server config names, when there is one). `asset/f -> target`: write the registry asset there.
        `~/.cache/huggingface`: nothing, the server fetches its own weights. Returns [] when the
        contract needs no help.
        """
        steps = []
        subdir = ""
        for mount in self.contract.get("mounts") or []:
            source = mount.get("source") or ""
            if source.startswith("asset/"):
                text = (REG / source).read_text(encoding="utf-8")
                match = re.search(r"^\s*model_name:\s*(\S+)", text, re.MULTILINE)
                if match:
                    subdir = match.group(1).strip("\"'")
                    text = re.sub(r"^(\s*host:\s*)127\.0\.0\.1", r"\g<1>0.0.0.0", text, flags=re.MULTILINE)
                steps.append(("asset", mount["target"], text))
        for mount in self.contract.get("mounts") or []:
            source = mount.get("source") or ""
            if source.startswith("${MODEL_ROOT}/"):
                provision = mount.get("provision") or {}
                repo = provision.get("repository") or self.instance.get("repository")
                revision = provision.get("revision") or self.instance.get("revision")
                target = mount["target"].rstrip("/")
                if not provision and subdir:
                    target = f"{target}/{subdir}"
                files = mount.get("files") or provision.get("files")
                steps.append(("weights", target, (repo, revision, files)))
        return steps

    def onstart_script(self):
        """One /bin/sh line: provision, then exec the contract's entrypoint and arguments."""
        import base64
        import shlex
        python = self.entrypoint if self.entrypoint and re.search(r"python3?$", self.entrypoint) else "python3"
        parts = []
        for kind, target, payload in self.provision:
            if kind == "asset":
                encoded = base64.b64encode(payload.encode()).decode()
                parts.append(f"mkdir -p {shlex.quote(os.path.dirname(target))} && printf %s {encoded} | base64 -d > {shlex.quote(target)}")
            else:
                repo, revision, files = (list(payload) + [None])[:3]  # a 2-tuple from an older caller means the whole revision
                patterns = [p.strip() for p in files.split(",") if p.strip()] if isinstance(files, str) else list(files or [])
                allow = f", allow_patterns={patterns!r}" if patterns else ""
                code = f"from huggingface_hub import snapshot_download as d; d({repo!r}, revision={revision!r}, local_dir={target!r}{allow})"
                # some engine images (the official llama.cpp CUDA builds) ship python3 with no pip, so
                # `pip install huggingface_hub` cannot run; the download then falls back to curl (present in the
                # image), which retries a dropped transfer and resumes it instead of starting over
                fallback = ("import fnmatch,json,os,subprocess,urllib.request\n"
                            f"repo,rev,target={repo!r},{revision!r},{target!r}\n"
                            f"patterns={patterns!r}\n"
                            "os.makedirs(target,exist_ok=True)\n"
                            "tree=json.load(urllib.request.urlopen('https://huggingface.co/api/models/%s/tree/%s'%(repo,rev)))\n"
                            "files=[s for s in tree if s.get('type')=='file' and (not patterns or any(fnmatch.fnmatch(s['path'],p) for p in patterns))]\n"
                            "if patterns and not files: raise SystemExit('nothing in the revision matches %r' % (patterns,))\n"
                            "for s in files:\n"
                            " path=s['path'];dest=os.path.join(target,path)\n"
                            " if os.path.exists(dest) and os.path.getsize(dest)==s.get('size'): continue\n"
                            " print('downloading',path,s.get('size'),'bytes',flush=True)\n"
                            " subprocess.check_call(['curl','-L','--fail','--retry','8','--retry-all-errors','--retry-delay','5','-C','-','--create-dirs','-o',dest,'https://huggingface.co/%s/resolve/%s/%s'%(repo,rev,path)])\n"
                            " if s.get('size') is not None and os.path.getsize(dest)!=s['size']:\n"
                            "  raise SystemExit('short download: '+path)\n")
                parts.append(f"mkdir -p {shlex.quote(target)}")
                hub = (f"({shlex.join([python, '-c', 'import huggingface_hub'])} || "
                       f"{shlex.join([python, '-m', 'pip', 'install', '-q', 'huggingface_hub'])}) && "
                       f"{shlex.join([python, '-c', code])}")
                parts.append(f"({hub} || {shlex.join([python, '-c', fallback])})")
        command = [self.entrypoint] if self.entrypoint else []
        parts.append("exec " + shlex.join(command + self.arguments))
        return " && ".join(parts)

    def shown(self):
        return {"image": self.image, "entrypoint": self.entrypoint, "arguments": self.arguments, "port": self.port,
                "disk": self.disk, "env": {k: ("<set>" if k == "HF_TOKEN" else v) for k, v in self.env.items()}}


# ----------------------------------------------------------------------------- RunPod
class RunPod:
    API = "https://rest.runpod.io/v1"
    name = "runpod"

    def __init__(self, args):
        self.cloud = args.cloud

    @staticmethod
    def key():
        key = os.environ.get("RUNPOD_API_KEY")
        if key:
            return key
        cfg = Path.home() / ".runpod" / "config.toml"
        if cfg.exists():
            match = re.search(r'apikey\s*=\s*"([^"]+)"', cfg.read_text(encoding="utf-8"), re.IGNORECASE)
            if match:
                return match.group(1)
        raise SystemExit("no RunPod API key: set RUNPOD_API_KEY or run `runpodctl config`")

    def call(self, method, path, body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.API}{path}", data=data, method=method, headers={
            "Authorization": f"Bearer {self.key()}", "Content-Type": "application/json",
            "User-Agent": "local-ai-lab/1"})  # Cloudflare in front of RunPod blocks the urllib default (error 1010)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise SystemExit(f"RunPod {method} {path} failed: {error.code} {error.read().decode(errors='replace')[:500]}")
        return json.loads(raw) if raw else None

    def gpu_ids(self, spec):
        if spec.gpu_override:
            return [spec.gpu_override]
        ids = (GPUS.get(spec.hardware_id) or {}).get("runpod")
        if not ids:
            raise SystemExit(f"no RunPod GPU mapping for {spec.hardware_id}; pass --gpu")
        return ids

    def describe(self, spec):
        return f"{self.gpu_ids(spec)[0]} ({self.cloud})"

    def create(self, spec, exclude):
        body = {
            "name": spec.name, "imageName": spec.image,
            "dockerEntrypoint": [spec.entrypoint] if spec.entrypoint else [],
            "dockerStartCmd": spec.arguments, "env": spec.env, "ports": [f"{spec.port}/http"],
            "containerDiskInGb": spec.disk, "volumeInGb": 0, "gpuTypeIds": self.gpu_ids(spec),
            "gpuTypePriority": "custom", "gpuCount": 1, "interruptible": False,
        }
        for cloud in dict.fromkeys([self.cloud, "SECURE", "COMMUNITY"]):
            body["cloudType"] = cloud
            try:
                pod = self.call("POST", "/pods", body)
                self.cloud = cloud
                return {"id": pod["id"], "cost": pod.get("costPerHr"), "gpu": self.gpu_ids(spec)[0],
                        "endpoint": f"https://{pod['id']}-{spec.port}.proxy.runpod.net"}
            except SystemExit as error:
                if "no instances currently available" not in str(error):
                    raise
                log(f"no {cloud} instances for {self.gpu_ids(spec)[0]} right now")
        raise SystemExit(f"no RunPod instances available for {self.gpu_ids(spec)[0]}; try later or lower --disk")

    def poll(self, handle):
        state = self.call("GET", f"/pods/{handle['id']}")
        status = state.get("desiredStatus")
        if status in ("EXITED", "TERMINATED"):
            raise SystemExit(f"pod {status} before the server became healthy")
        started = bool((state.get("runtime") or {}).get("uptimeInSeconds"))
        handle["gpu"] = (state.get("machine") or {}).get("gpuTypeId") or handle["gpu"]
        return started, f"pod {status}"

    def destroy(self, handle):
        self.call("DELETE", f"/pods/{handle['id']}")


# ----------------------------------------------------------------------------- Vast.ai
class Vast:
    name = "vast"

    def __init__(self, args):
        if not shutil.which("vastai"):
            raise SystemExit("vastai CLI not found: `uv tool install vastai` and put the key in ~/.config/vastai/vast_api_key")
        self.min_inet = args.vast_min_inet
        self.verified = "any" if getattr(args, "vast_any_host", False) else "true"  # unverified hosts: for cards verified hosts never offer
        self.min_cuda = args.vast_min_cuda
        self.max_price = args.vast_max_price
        # an offload recipe keeps experts in host RAM and reads tables from NVMe: rent only a host that has them
        self.min_ram = getattr(args, "vast_min_ram", 0) or 0
        self.min_disk_bw = getattr(args, "vast_min_disk_bw", 0) or 0
        self.min_pcie = getattr(args, "vast_min_pcie", 0) or 0
        # hosts bill the download per GB; at $0.04/GB a 70 GB image and weights pull costs more than hours of compute
        self.max_inet_cost = getattr(args, "vast_max_inet_cost", 0.02)

    @staticmethod
    def cli(*argv, timeout=120):
        # `destroy` asks for confirmation on stdin; `create` prints "Started. {python dict}" even with --raw
        # --raw must precede --args: everything after --args is passed to the container
        result = subprocess.run(["vastai", *argv[:2], "--raw", *argv[2:]], capture_output=True, text=True, timeout=timeout, input="y\n")
        text = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0 or (not text and '"error": true' in err):
            raise SystemExit(f"vastai {' '.join(argv[:2])} failed: {(err or text)[:500]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start >= 0:
            import ast
            try:
                return ast.literal_eval(text[start:])
            except (ValueError, SyntaxError):
                pass
        if argv[0] == "destroy":
            return {"success": "destroying" in text}
        raise SystemExit(f"vastai {' '.join(argv[:2])}: unexpected output: {text[:300]}")

    def gpu_name(self, spec):
        if spec.gpu_override:
            return spec.gpu_override
        name = (GPUS.get(spec.hardware_id) or {}).get("vast")
        if not name:
            raise SystemExit(f"no Vast GPU mapping for {spec.hardware_id}; pass --gpu")
        return name

    def describe(self, spec):
        return f"{self.gpu_name(spec)} (vast)"

    def offers(self, spec, exclude):
        name = self.gpu_name(spec).replace(" ", "_")
        # query filter is in GB (results report MB); bound both sides so an 8 GB recipe never runs on the 16 GB variant of the same name
        lo, hi = int(max(spec.vram_gb - 1, 1)), int(spec.vram_gb + 1)
        # cuda_max_good is the host driver's CUDA ceiling; the image's NVIDIA_REQUIRE_CUDA floor must be at or below it
        # (12.9 for the pinned sglang cu12 image; 13.0 for vllm-openai and sglang dev-cu13; 13.2 for tabbyapi cu13 and
        # the attested tabbyapi-exl3). Older drivers cannot init the image (12.4 hosts fail llama.cpp too).
        query = (f"num_gpus=1 rentable=true verified={self.verified} gpu_name={name} gpu_ram>={lo} "  # the <= side is applied below: the server reads it in MB
                 f"inet_down>={self.min_inet} disk_space>={spec.disk + 5} reliability>0.9 cuda_max_good>={self.min_cuda} "
                 f"dph_total<={self.max_price} "  # the campaign ceiling; a card with no offer under it is reported, never rented above it
                 + (f"cpu_ram>={self.min_ram} " if self.min_ram else "")  # GB, like gpu_ram
                 + (f"disk_bw>={self.min_disk_bw} " if self.min_disk_bw else "")  # MB/s
                 + (f"pcie_bw>={self.min_pcie} " if self.min_pcie else "")  # GB/s host-to-device: offloaded experts cross it every token
                 + (f"inet_down_cost<={self.max_inet_cost} " if self.max_inet_cost else "")  # $/GB downloaded
                 + "geolocation notin [CN]")  # hosts that cannot reach Hugging Face never finish the weights download
        offers = self.cli("search", "offers", query, "-o", "dph_total")
        offers = [o for o in offers if o.get("id") not in exclude and o.get("machine_id") not in exclude
                  and (o.get("gpu_ram") or 0) <= hi * 1024]  # results report MB; keep the 8 GB variant off a 16 GB card and vice versa
        if not offers:
            raise SystemExit(f"no Vast offers for {query}")
        # a run's cost: about two hours of compute plus the pull (image and weights, at most the disk we ask for)
        est = lambda o: 2 * (o.get("dph_total") or 0) + (o.get("inet_down_cost") or 0) * spec.disk
        offers.sort(key=est)
        # cheapest hosts are often the slowest pullers: within 1.5x of the best estimate, take the fastest downlink first
        cap = est(offers[0]) * 1.5
        cheap = [o for o in offers if est(o) <= cap]
        cheap.sort(key=lambda o: -(o.get("inet_down") or 0))
        return cheap + [o for o in offers if o not in cheap]

    def create(self, spec, exclude):
        env = f"-p {spec.port}:{spec.port} " + " ".join(f"-e {k}={v}" for k, v in spec.env.items())
        for offer in self.offers(spec, exclude)[:5]:  # search results go stale; try the next cheapest
            argv = ["create", "instance", str(offer["id"]), "--image", spec.image, "--disk", str(spec.disk),
                    "--env", env, "--label", spec.name, "--cancel-unavail"]
            if spec.provision:
                # the plugin bind-mounts weights and config; here the container provisions them, then execs the contract
                if not spec.entrypoint:
                    raise SystemExit("in-container provisioning needs an explicit entrypoint in the contract")
                # the onstart override is exec'd as one argv token, so the shell must be the entrypoint
                argv += ["--onstart-cmd", "/bin/sh", "--args", "-c", spec.onstart_script()]
            else:
                if spec.entrypoint:
                    argv += ["--entrypoint", spec.entrypoint]
                if spec.arguments:
                    argv += ["--args", *spec.arguments]
            try:
                result = self.cli(*argv)
            except SystemExit as error:
                if "no_such_ask" in str(error) or "not available" in str(error):
                    log(f"offer {offer['id']} gone; trying the next one")
                    exclude.add(offer["id"])
                    continue
                raise
            if not result.get("success"):
                raise SystemExit(f"vast create failed: {result}")
            return {"id": result["new_contract"], "offer": offer["id"], "machine": offer.get("machine_id"),
                    "cost": offer.get("dph_total"), "gpu": offer.get("gpu_name"), "host": offer.get("geolocation"), "endpoint": None}
        raise SystemExit("every Vast offer tried was already gone; retry in a minute")

    def poll(self, handle):
        info = self.cli("show", "instance", str(handle["id"]))
        status = info.get("actual_status") or info.get("cur_state") or "?"
        full = (info.get("status_msg") or "").strip()
        if full and full != handle.get("last_msg"):
            handle["last_msg"] = full
            log(f"vast status: {full[:600]}")
        msg = full.splitlines()
        msg = msg[-1][:80] if msg else ""
        if status in ("exited", "stopped", "error") or (info.get("intended_status") == "stopped"):
            raise SystemExit(f"vast instance {status}: {msg}")
        ports = (info.get("ports") or {}).get(f"{handle['port']}/tcp") or []
        ip = info.get("public_ipaddr")
        if ports and ip:
            handle["endpoint"] = f"http://{ip}:{ports[0]['HostPort']}"
        started = status == "running"
        return started, f"vast {status} {msg}".strip()

    LISTENING = ("listening on http", "uvicorn running on", "application startup complete", "the server is fired up")

    def listening(self, handle):
        """True once the container log says the server is bound; only then does an unreachable port mean a firewalled host."""
        try:
            result = subprocess.run(["vastai", "logs", str(handle["id"]), "--tail", "400"], capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return False
        text = (result.stdout or "").lower()
        return any(marker in text for marker in self.LISTENING)

    def destroy(self, handle):
        self.cli("destroy", "instance", str(handle["id"]))


PROVIDERS = {"runpod": RunPod, "vast": Vast}


# ----------------------------------------------------------------------------- recipe handling
def load_recipe(recipe_id, revalidate=False):
    path = REG / "recipe" / f"{recipe_id}.json"
    if not path.exists():
        raise SystemExit(f"no recipe {recipe_id}")
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if recipe.get("status") != "candidate" and not revalidate:
        raise SystemExit(f"only candidates can be validated; {recipe_id} is {recipe.get('status')} (pass --revalidate)")
    launch = recipe.get("launch") or {}
    contract = launch if launch.get("kind") == "docker" else recipe.get("draft_launch")
    if not contract:
        raise SystemExit(f"{recipe_id} has neither a docker launch nor a draft_launch")
    if not re.search(r"@sha256:[0-9a-f]{64}$", contract.get("image", "")):
        raise SystemExit(f"{recipe_id}: image is not digest-pinned")
    return path, recipe, contract


def strip_host_ipc(path, recipe):
    """The rented container runs without host IPC; a recipe validated here must not claim to need it."""
    changed = False
    for key in ("launch", "draft_launch"):
        block = recipe.get(key)
        if isinstance(block, dict) and block.get("ipc") == "host":
            del block["ipc"]
            changed = True
    if changed:
        path.write_text(json.dumps(recipe, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        log("dropped ipc=host from the contract: validated without it")


# ----------------------------------------------------------------------------- main
def run_once(provider, spec, path, recipe, args, attempt, exclude):
    log(f"attempt {attempt}: renting {provider.describe(spec)}, image {spec.image.split('@')[0]}")
    handle = provider.create(spec, exclude)
    handle["port"] = spec.port
    started_at = time.monotonic()
    log(f"{provider.name} {handle['id']} created, {handle.get('cost')} /h, gpu {handle.get('gpu')}")
    container_up = False
    timeouts = 0  # consecutive probes that timed out after the server was bound: a firewalled host
    bound = False
    last_listen_check = float("-inf")
    try:
        deadline = started_at + args.timeout
        last_note = 0
        while True:
            endpoint = handle.get("endpoint")
            probe = http_probe(f"{endpoint}/v1/models") if endpoint else "no-endpoint"
            if probe == "ok":
                break
            if time.monotonic() > deadline:
                raise SystemExit(f"server did not become healthy within {args.timeout}s")
            started, note = provider.poll(handle)
            container_up = container_up or started
            elapsed = int(time.monotonic() - started_at)
            if not started and elapsed > args.start_timeout:  # current status, not latched: a host stuck on the pull stays "loading"
                exclude.update({handle.get("offer"), handle.get("machine")})
                raise StartStalled(f"container not started after {elapsed}s ({note})")
            # a closed container port drops packets on some hosts, so a timeout only means "firewalled"
            # once the server has logged that it is listening (checked at most every 60s)
            if container_up and probe == "timeout":
                if time.monotonic() - last_listen_check >= 60:
                    bound = getattr(provider, "listening", lambda h: False)(handle)
                    last_listen_check = time.monotonic()
                timeouts = timeouts + 1 if bound else 0
            else:
                timeouts = 0
            if timeouts * 10 > args.reach_timeout:
                exclude.update({handle.get("offer"), handle.get("machine")})
                raise StartStalled(f"port unreachable for {args.reach_timeout}s while the container ran ({note})")
            if elapsed - last_note >= 120:
                log(f"waiting for /v1/models ({elapsed}s, {note}, container {'up' if container_up else 'not started'})")
                last_note = elapsed
            time.sleep(10)
        healthy = int(time.monotonic() - started_at)
        endpoint = handle["endpoint"]
        log(f"server healthy after {healthy}s at {endpoint}")

        strip_host_ipc(path, recipe)
        harness = f"validate_rented.py on {provider.name} {handle.get('gpu')}"
        if spec.provision:
            harness += " (weights and config materialized in-container in place of the bind mounts)"
        gateway = os.environ.get("LOCAL_AI_GATEWAY") or str(Path.home() / "local-registry" / "local-ai-images" / "gateway" / "gateway.py")
        accept_argv = [sys.executable, str(ROOT / "scripts" / "accept_recipe.py"), recipe["id"], "--endpoint", endpoint, "--harness", harness]
        if args.revalidate:
            accept_argv.append("--revalidate")
        if Path(gateway).exists():
            accept_argv += ["--gateway", gateway]
        else:
            log("no local gateway.py found; dialects will not be recorded (set LOCAL_AI_GATEWAY)")
        result = subprocess.run(accept_argv, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit("acceptance failed; recipe stays candidate")
        if args.recommend:
            promoted = json.loads(path.read_text(encoding="utf-8"))
            promoted["recommended"] = True
            path.write_text(json.dumps(promoted, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"PROMOTED {recipe['id']} (validated on {provider.name} {handle.get('gpu')}){' and flagged recommended' if args.recommend else ''}")
        return 0
    finally:
        elapsed_h = (time.monotonic() - started_at) / 3600
        if args.keep:
            log(f"keeping {provider.name} {handle['id']} ({handle.get('endpoint')}); destroy it yourself")
        else:
            try:
                provider.destroy(handle)
                log(f"{provider.name} {handle['id']} destroyed")
            except SystemExit as error:
                log(f"WARNING: could not destroy {provider.name} {handle['id']}: {error}")
        if handle.get("cost"):
            log(f"approx cost: {handle['cost'] * elapsed_h:.2f} ({elapsed_h * 60:.0f} min at {handle['cost']:.3f}/h)")


def main():
    import signal

    def terminate(*_):  # background jobs ignore SIGINT; SIGTERM must unwind the finally blocks so the box is destroyed
        raise SystemExit("terminated")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe_id")
    parser.add_argument("--provider", default="vast", choices=sorted(PROVIDERS))
    parser.add_argument("--gpu", help="provider GPU name; default from the recipe's hardware_id")
    parser.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"], help="runpod only")
    parser.add_argument("--vast-min-inet", type=int, default=500, help="vast only: minimum host downlink in Mbps")
    parser.add_argument("--vast-max-price", type=float, default=1.0, help="vast only: maximum offer price in $/h (dph_total)")
    parser.add_argument("--vast-min-cuda", type=float, default=12.9,
                        help="vast only: minimum host CUDA (cuda_max_good); pass the image's NVIDIA_REQUIRE_CUDA floor, e.g. 13.2 for tabbyapi cu13")
    parser.add_argument("--disk", type=int, default=50, help="container disk in GB (image + weights)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds to wait for /v1/models")
    parser.add_argument("--start-timeout", type=int, default=600, help="seconds to wait for the container to start before retrying elsewhere")
    parser.add_argument("--retries", type=int, default=1, help="fresh-host retries when the container never starts")
    parser.add_argument("--reach-timeout", type=int, default=180, help="seconds of port timeouts, with the container running, before treating the host as firewalled")
    parser.add_argument("--keep", action="store_true", help="leave the box running after acceptance")
    parser.add_argument("--recommend", action="store_true", help="flag the promoted recipe recommended for its hardware id")
    parser.add_argument("--dry-run", action="store_true", help="print the container spec and exit")
    parser.add_argument("--revalidate", action="store_true", help="re-run acceptance on a validated recipe (refreshes evidence and dialects)")
    args = parser.parse_args()

    path, recipe, contract = load_recipe(args.recipe_id, args.revalidate)
    spec = Spec(recipe, contract, args)
    provider = PROVIDERS[args.provider](args)
    if args.dry_run:
        print(json.dumps({"provider": provider.describe(spec), **spec.shown()}, indent=2))
        return 0

    exclude = set()
    attempts = 1 + args.retries
    for attempt in range(1, attempts + 1):
        try:
            return run_once(provider, spec, path, recipe, args, attempt, exclude)
        except StartStalled as stall:
            log(f"attempt {attempt}: {stall}")
            if attempt == attempts:
                raise SystemExit("container never started; giving up")
            log("retrying on a different host")
    return 1


if __name__ == "__main__":
    sys.exit(main())
