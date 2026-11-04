#!/usr/bin/env python3
"""The Monday digest: what is waiting on us across the Local AI repos, and whether every pin still resolves.

    lab/digest.py              # post it as one comment on Linear HOM-174 (needs LINEAR_API_KEY)
    lab/digest.py --dry-run    # print it instead

Report only: it reads GitHub (GITHUB_TOKEN or GH_TOKEN), the site, ghcr.io and huggingface.co anonymously,
and changes nothing. A section that cannot be read says so instead of stopping the digest. Standard library only.
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ME = "0xSero"
OWNER = "sybil-solutions"
REPOS = ['ai-data-extraction', 'codex-shim', 'deepseek-v4.1-flash-4x-rtx-pro-6000', 'deepseek-v4.1-flash-h200', 'dsv41-flash-offload', 'glm-flash-lite', 'local-ai-images', 'local-ai-recipe-kit', 'local-ai-registry', 'local-studio', 'moet', 'moetier', 'omarchy-local-ai', 'qwen-serve', 'sglang-exl3', 'sovereign-trellis', 'trellis-serve']
PLUGIN = "omarchy-local-ai"
MARKETPLACE = "omacom/omarchy-plugin-marketplace/issues/10396"
UPSTREAM, UPSTREAM_PR = "omacom/omarchy", 13036
SITE = "https://local.sybilsolutions.ai"
LINEAR_ISSUE = "HOM-174"
WAIT_DAYS = 3
NOW = dt.datetime.now(dt.timezone.utc)
MANIFESTS = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                       "application/vnd.docker.distribution.manifest.list.v2+json",
                       "application/vnd.docker.distribution.manifest.v2+json"])
flags = []


def flag(text):
    flags.append(text)
    return f"**{text}**"


def fetch(url, headers=None, method="GET", body=None, timeout=30):
    """(status, parsed JSON or text). HTTP errors come back as their status; a network failure is status 0."""
    req = urllib.request.Request(url, data=body, method=method, headers={"User-Agent": "local-ai-digest", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except OSError as e:  # refused, DNS, timeout
        return 0, str(e)
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, raw.decode(errors="replace")


def gh(path):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    status, data = fetch(f"https://api.github.com/{path}", headers)
    if status != 200:
        raise LookupError(f"GitHub {path.split('?')[0]}: HTTP {status}")
    return data


def days(ts):
    return max(0.0, (NOW - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 86400)


def readable():
    """{repo: default branch} for the repos this token can read, and a note for each it cannot."""
    found, notes = {}, []
    for repo in REPOS:
        try:
            found[repo] = gh(f"repos/{OWNER}/{repo}")["default_branch"]
        except LookupError as e:
            notes.append(f"- {repo}: skipped, not readable with this token ({e})")
    return found, notes


# ----------------------------------------------------------------------------- sections
def community(repos):
    lines = []
    for repo in repos:
        full = f"{OWNER}/{repo}"
        for it in gh(f"repos/{full}/issues?state=open&per_page=100"):
            who = it["user"]["login"]
            if it["user"]["type"] == "Bot":  # dependabot and other bots
                continue
            n, pr = it["number"], "pull_request" in it
            said = [it] + gh(f"repos/{full}/issues/{n}/comments?per_page=100")
            if pr:
                said += gh(f"repos/{full}/pulls/{n}/comments?per_page=100")
                said += [{**r, "created_at": r["submitted_at"]} for r in gh(f"repos/{full}/pulls/{n}/reviews?per_page=100")
                         if r.get("submitted_at")]
            said = [s for s in said if s.get("user") and s["user"]["type"] != "Bot"]
            theirs = max((s["created_at"] for s in said if s["user"]["login"] != ME), default=it["created_at"])
            wait = days(theirs)
            if who == ME and not any(s["user"]["login"] != ME for s in said):
                state = "owner delivery item, see linked release evidence"
            elif any(s["user"]["login"] == ME and s["created_at"] > theirs for s in said):
                state = "0xSero replied last"
            elif wait > WAIT_DAYS:
                state = flag(f"waiting {wait:.1f}d for 0xSero")
            else:
                state = f"waiting {wait:.1f}d for 0xSero"
            kind = "PR" if pr else "issue"
            lines.append(f"- [{repo}#{n}]({it['html_url']}) {kind} by @{who}, open {days(it['created_at']):.0f}d: "
                         f"{it['title'][:80]} ({state})")
        runs = gh(f"repos/{full}/actions/runs?status=action_required&per_page=100")["workflow_runs"]
        branches = {}  # newest run first, so each branch links its newest waiting run
        for run in runs:
            branches.setdefault((run["head_branch"], run["actor"]["login"]), []).append(run)
        if runs:
            waiting = ", ".join(f"[{b}]({rs[0]['html_url']}) by @{a} ({len(rs)}, newest {days(rs[0]['created_at']):.0f}d)"
                                for (b, a), rs in branches.items())
            lines.append(f"- {repo}: {flag(f'{len(runs)} PR workflow runs in action_required')}: {waiting}")
    return lines or ["- no open delivery or community items"]


def ci(repos):
    lines = []
    for repo, branch in repos.items():
        latest = {}  # workflow name -> its newest completed run on the default branch
        for run in gh(f"repos/{OWNER}/{repo}/actions/runs?branch={branch}&status=completed&per_page=50")["workflow_runs"]:
            latest.setdefault(run["name"], run)
        bad = [r for r in latest.values() if r["conclusion"] not in ("success", "skipped", "neutral")]
        if not latest:
            lines.append(f"- {repo} ({branch}): no completed runs")
        elif not bad:
            lines.append(f"- {repo} ({branch}): ok ({', '.join(sorted(latest))})")
        for r in bad:
            what = flag(f"{r['name']} {r['conclusion']}")
            lines.append(f"- {repo} ({branch}): {what} at [{r['head_sha'][:7]}]({r['html_url']}), {days(r['updated_at']):.1f}d ago")
    return lines


def releases():
    rel = gh(f"repos/{OWNER}/{PLUGIN}/releases/latest")
    tag = rel["tag_name"]
    sha = gh(f"repos/{OWNER}/{PLUGIN}/commits/{tag}")["sha"]
    issue = gh(f"repos/{MARKETPLACE}")
    target = re.search(r"### Target commit\s+([0-9a-f]{7,40})", issue["body"] or "")
    market = f"[marketplace #{issue['number']}]({issue['html_url']}) ({issue['state']})"
    lines = [f"- {PLUGIN} latest release [{tag}]({rel['html_url']}) at {sha[:7]}, published {days(rel['published_at']):.1f}d ago"]
    if not target:
        lines.append(f"- {market}: {flag('no target commit in its body')}")
    elif sha.startswith(target[1]):
        lines.append(f"- {market} verifies {target[1][:7]}, the latest release")
    else:
        behind = gh(f"repos/{OWNER}/{PLUGIN}/compare/{target[1]}...{sha}")["ahead_by"]
        lines.append(f"- {market} verifies {target[1][:7]}: {flag(f'{behind} commits behind {tag}')}")
    return lines


def upstream():
    pr = gh(f"repos/{UPSTREAM}/pulls/{UPSTREAM_PR}")
    if pr["mergeable"] is None:  # GitHub computes it on the first read
        time.sleep(5)
        pr = gh(f"repos/{UPSTREAM}/pulls/{UPSTREAM_PR}")
    cmp = gh(f"repos/{UPSTREAM}/compare/{pr['base']['ref']}...{pr['head']['label']}")
    state = "merged" if pr["merged"] else pr["state"]
    mergeable = f"mergeable={pr['mergeable']}, mergeable_state={pr['mergeable_state']}"
    if pr["state"] == "open" and pr["mergeable_state"] != "clean":
        mergeable = flag(mergeable)
    return [f"- [{UPSTREAM}#{UPSTREAM_PR}]({pr['html_url']}) {state}, {mergeable}, updated {days(pr['updated_at']):.1f}d ago",
            f"- {pr['head']['label']} is {cmp['behind_by']} commits behind and {cmp['ahead_by']} ahead of "
            f"{UPSTREAM.split('/')[0]}:{pr['base']['ref']} ({cmp['status']})"]


def site():
    status, _ = fetch(SITE)
    lines = [f"- {SITE}: HTTP {status}" if status == 200 else f"- {SITE}: {flag(f'HTTP {status}')}"]
    status, live = fetch(f"{SITE}/api/v2/catalog.json")
    if status != 200 or not isinstance(live, dict):
        return lines + [f"- /api/v2/catalog.json: {flag(f'HTTP {status}')}"]
    main = json.loads((ROOT / "dist" / "catalog.json").read_text(encoding="utf-8"))
    counts = f"{len(live['cards'])} cards, {len(live['recipes'])} recipes"
    if live.get("sha256") == main.get("sha256"):
        return lines + [f"- /api/v2/catalog.json: {counts}, same as main's dist/catalog.json"]
    return lines + [f"- /api/v2/catalog.json: {counts}; {flag('not the catalog on main')} "
                    f"({len(main['cards'])} cards, {len(main['recipes'])} recipes in main's dist/catalog.json)"]


# ----------------------------------------------------------------------------- recipe drift
def pins():
    """Every historical or org GHCR image digest in registry/ and the plugin export, and every Hugging Face weights pin in registry/."""
    images, weights = set(), set()
    for f in sorted([*(ROOT / "registry").rglob("*.json"), ROOT / "plugin" / "v2" / "recipes.json"]):
        text = f.read_text(encoding="utf-8")
        images |= set(re.findall(r"ghcr\.io/((?:0xsero|sybil-solutions)/[\w.-]+?)(?::[\w.-]+)?@sha256:([0-9a-f]{64})", text))
        if f.parts[-2] == "v2":
            continue

        def visit(v):
            if isinstance(v, dict):
                for k, c in v.items():
                    if k == "weights" and isinstance(c, str) and "@" in c:  # a recipe: repo@commit
                        weights.add(tuple(c.split("@")))
                    elif k == "weights" and isinstance(c, list):  # a launch: [{repo, revision}]
                        weights.update((w["repo"], w.get("revision")) for w in c if isinstance(w, dict) and "repo" in w)
                    else:
                        visit(c)
            elif isinstance(v, list):
                for c in v:
                    visit(c)
        visit(json.loads(text))
    return images, weights


def retry(check):
    def run(pin):
        result = check(pin)
        if result and (result[0] in (0, 429) or result[0] >= 500):  # transient: once more
            time.sleep(5)
            result = check(pin)
        return pin, result
    return run


def image_missing(pin):
    name, digest = pin
    status, token = fetch(f"https://ghcr.io/token?scope=repository:{name}:pull")
    if status != 200:
        return status, "no anonymous pull token (private or deleted package)"
    status, _ = fetch(f"https://ghcr.io/v2/{name}/manifests/sha256:{digest}", method="HEAD",
                      headers={"Authorization": f"Bearer {token['token']}", "Accept": MANIFESTS})
    return None if status == 200 else (status, "manifest not found")


def weights_missing(pin):
    repo, rev = pin
    status, _ = fetch(f"https://huggingface.co/api/models/{repo}" + (f"/revision/{rev}" if rev else ""))
    if status == 200:
        return None
    return status, {401: "repo missing or private", 404: "revision gone"}.get(status, "unreadable")


def drift():
    images, weights = pins()
    jobs = [(retry(image_missing), p) for p in sorted(images)] + [(retry(weights_missing), p) for p in sorted(weights)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: job[0](job[1]), jobs))
    bad_images = [(p, r) for p, r in results[:len(images)] if r]
    bad_weights = [(p, r) for p, r in results[len(images):] if r]
    lines = [f"- images: {len(images) - len(bad_images)}/{len(images)} GHCR digests resolve",
             f"- weights: {len(weights) - len(bad_weights)}/{len(weights)} Hugging Face pins resolve"]
    for (name, digest), (status, why) in bad_images:
        lines.append(f"  - {flag(f'ghcr.io/{name}@sha256:{digest[:12]}')}: HTTP {status}, {why}")
    for (repo, rev), (status, why) in bad_weights:
        at = f"@{rev[:12]}" if rev else ""
        lines.append(f"  - {flag(f'{repo}{at}')} ([hf](https://huggingface.co/{repo})): HTTP {status}, {why}")
    return lines


# ----------------------------------------------------------------------------- the digest
def section(title, make, *args):
    try:
        lines = make(*args)
    except Exception as e:  # one unreadable source must not cost the rest of the digest
        lines = [f"- {flag('section failed')}: {type(e).__name__}: {e}"]
    return [f"### {title}", *lines, ""]


def digest():
    repos, skipped = readable()
    body = [*section("Open delivery and community items", lambda: community(repos) + skipped),
            *section("CI (latest default-branch run per workflow)", ci, repos),
            *section("Releases", releases),
            *section("Upstream", upstream),
            *section("Site", site),
            *section("Recipe drift", drift)]
    head = f"## Local AI digest, {NOW:%Y-%m-%d}\n\n{len(flags)} flagged" + (": " + "; ".join(flags[:8]) if flags else "")
    if len(flags) > 8:
        head += f"; and {len(flags) - 8} more"
    return head + "\n\n" + "\n".join(body).rstrip() + "\n\n_Report only, from `lab/digest.py` in local-ai-registry._\n"


def post(body):
    key = os.environ.get("LINEAR_API_KEY")
    if not key:
        raise SystemExit("LINEAR_API_KEY is not set (use --dry-run to print the digest)")
    query = "mutation($input: CommentCreateInput!) { commentCreate(input: $input) { success comment { id url } } }"
    payload = json.dumps({"query": query, "variables": {"input": {"issueId": LINEAR_ISSUE, "body": body}}}).encode()
    status, data = fetch("https://api.linear.app/graphql", {"Authorization": key, "Content-Type": "application/json"},
                         "POST", payload)
    made = isinstance(data, dict) and (data.get("data") or {}).get("commentCreate") or {}
    if status != 200 or not made.get("success"):
        errors = data.get("errors") if isinstance(data, dict) else data
        raise SystemExit(f"Linear commentCreate failed: HTTP {status} {json.dumps(errors)[:500]}")
    print(f"posted to {LINEAR_ISSUE}: {made['comment']['url']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the markdown instead of posting it")
    args = ap.parse_args()
    body = digest()
    if args.dry_run:
        print(body)
    else:
        print(body, file=sys.stderr)
        post(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
