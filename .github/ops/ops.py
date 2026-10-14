#!/usr/bin/env python3
"""The project's open work, for the dashboard at usage.sybilsolutions.ai and the bounty ledger issue.

Writes ops.json (read by the dashboard from the `ops` branch) and ledger.md (the body of the pinned ledger issue).
Everything here is public: issue and PR titles, logins, counts, the marketplace catalog. Run hourly by ops.yml.

Bounty rules (BOUNTIES.md in sybil-solutions/.github): a bounty pays once, to the author of the merged PR that closes
it; one payout per person per device; at most $1,000 a month by merge month; a month is paid in the first week of the
next one.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.request

ORG = "sybil-solutions"
REGISTRY = f"{ORG}/local-ai-registry"
PLUGIN = f"{ORG}/omarchy-local-ai"
MARKET = "omacom/omarchy-plugin-marketplace"
PLUGIN_ID = "sero.local-ai"
VERIFY_ISSUE, MIGRATION_PR = 11207, 11412
CAP = 1000
OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def gh(*args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120, check=True).stdout


def ghjson(*args):
    return json.loads(gh(*args) or "null")


def checks(rollup):
    states = {(c.get("conclusion") or c.get("state") or c.get("status") or "").lower() for c in rollup or []} - {""}
    if not states:
        return "none"
    if states & {"failure", "error", "cancelled", "timed_out", "action_required"}:
        return "failing"
    if states <= {"success", "neutral", "skipped"}:
        return "passing"
    return "running"


def bounties(prs):
    issues = ghjson("issue", "list", "-R", REGISTRY, "--label", "bounty", "--state", "all", "--limit", "500",
                    "--json", "number,title,state,labels")
    out = []
    for i in sorted(issues, key=lambda x: x["number"]):
        n = i["number"]
        labels = [l["name"] for l in i["labels"]]
        amount = next((int(m.group(1)) for l in labels if (m := re.match(r"bounty: \$(\d+)", l))), 50)
        comments = ghjson("api", f"repos/{REGISTRY}/issues/{n}/comments?per_page=100")
        claims = [{"by": c["user"]["login"], "at": c["created_at"]} for c in comments
                  if re.match(r"\s*/attempt\b", c["body"] or "")]
        ref = re.compile(rf"(?<![\w/])#{n}\b|/issues/{n}\b")
        subs = sorted(({"pr": p["number"], "by": p["author"]["login"], "at": p["createdAt"],
                        "state": "merged" if p["mergedAt"] else p["state"].lower(), "mergedAt": p["mergedAt"],
                        "checks": checks(p["statusCheckRollup"])}
                       for p in prs if ref.search((p["title"] or "") + "\n" + (p["body"] or ""))), key=lambda s: s["at"])
        won = next((s for s in subs if s["state"] == "merged"), None)
        out.append({"issue": n, "device": re.sub(r"^Bounty \$\d+: recipe for ", "", i["title"]).strip(),
                    "amount": amount, "state": i["state"].lower(), "claims": claims, "prs": subs,
                    "paidTo": won and won["by"], "pr": won and won["pr"], "month": won and won["mergedAt"][:7]})
    return out


def market():
    issue = ghjson("issue", "view", str(VERIFY_ISSUE), "-R", MARKET, "--json", "state,labels,body,comments,updatedAt")
    target = re.search(r"### Target commit\s+([0-9a-f]{40})", issue["body"] or "")
    last = next((c for c in reversed(issue["comments"]) if c["author"]["login"] != "github-actions"), None)
    mig = ghjson("pr", "view", str(MIGRATION_PR), "-R", MARKET, "--json", "state,mergeable,updatedAt")
    listing = {}
    try:
        with urllib.request.urlopen("https://plugins.omarchy.org/catalog.json", timeout=30) as r:
            cat = json.load(r)
        p = next((x for x in (cat.get("plugins", cat) if isinstance(cat, dict) else cat) if x.get("id") == PLUGIN_ID), {})
        listing = {k: p.get(k) for k in ("version", "verificationStatus", "verificationCommit", "installAvailable", "repo")}
    except Exception as e:  # the catalog being down must not stop the rest
        listing = {"error": str(e)[:120]}
    return {"verify": {"issue": VERIFY_ISSUE, "state": issue["state"].lower(),
                       "labels": [l["name"] for l in issue["labels"]], "target": target and target.group(1),
                       "lastBy": last and last["author"]["login"], "lastAt": last and last["createdAt"]},
            "migration": {"pr": MIGRATION_PR, "state": mig["state"].lower(), "mergeable": mig["mergeable"].lower()},
            "listing": listing}


def repo(name):
    prs = ghjson("pr", "list", "-R", name, "--state", "open", "--limit", "200",
                 "--json", "number,title,author,isDraft,statusCheckRollup,createdAt")
    issues = ghjson("issue", "list", "-R", name, "--state", "open", "--limit", "500", "--json", "number,labels")
    return {"prs": len(prs), "drafts": sum(p["isDraft"] for p in prs),
            "passing": sum(checks(p["statusCheckRollup"]) == "passing" for p in prs if not p["isDraft"]),
            "failing": sum(checks(p["statusCheckRollup"]) == "failing" for p in prs if not p["isDraft"]),
            "issues": sum(not any(l["name"] in ("bounty", "bounty-ledger") for l in i["labels"]) for i in issues),
            "list": [{"n": p["number"], "title": p["title"][:90], "by": p["author"]["login"], "draft": p["isDraft"],
                      "checks": checks(p["statusCheckRollup"])} for p in prs]}


def main():
    now = dt.datetime.now(dt.timezone.utc)
    month = now.strftime("%Y-%m")
    all_prs = ghjson("pr", "list", "-R", REGISTRY, "--state", "all", "--limit", "1000",
                     "--json", "number,title,body,author,state,createdAt,mergedAt,statusCheckRollup")
    b = bounties(all_prs)
    months = {}
    for x in b:
        if x["month"]:
            m = months.setdefault(x["month"], {"accepted": 0, "owed": 0})
            m["accepted"] += 1
            m["owed"] += x["amount"]
    cur = months.get(month, {"accepted": 0, "owed": 0})
    plugin = repo(PLUGIN)
    plugin["devAhead"] = ghjson("api", f"repos/{PLUGIN}/compare/main...dev")["ahead_by"]
    plugin["version"] = json.loads(gh("api", f"repos/{PLUGIN}/contents/manifest.json", "-H",
                                      "Accept: application/vnd.github.raw"))["version"]
    cat = json.load(open("plugin/v3/recipes.json"))
    ops = {
        "updated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bounty": {"month": month, "cap": CAP, "owed": cur["owed"], "accepted": cur["accepted"],
                   "open": sum(x["state"] == "open" for x in b),
                   "openValue": sum(x["amount"] for x in b if x["state"] == "open"),
                   "claims": sum(len(x["claims"]) for x in b),
                   "prsOpen": sum(s["state"] == "open" for x in b for s in x["prs"]),
                   "months": months, "list": b},
        "plugin": plugin,
        "registry": repo(REGISTRY),
        "marketplace": market(),
        "devices": {"kinds": len(cat["hardware"]), "recipes": sum(len(h["recipes"]) for h in cat["hardware"].values())},
    }
    json.dump(ops, open(os.path.join(OUT, "ops.json"), "w"), indent=1)

    md = ["Recipe bounties: who claimed, who submitted, what was accepted. Updated every hour; rules in "
          "[BOUNTIES.md](https://github.com/sybil-solutions/.github/blob/main/BOUNTIES.md); dashboard at "
          "[usage.sybilsolutions.ai](https://usage.sybilsolutions.ai/#work).", "",
          f"**{month}:** {cur['accepted']} accepted, ${cur['owed']} of the ${CAP} monthly cap, ${CAP - cur['owed']} left. "
          "Payouts for a month go out in the first week of the next month.", "",
          "| Bounty | Device | $ | Claims (`/attempt`) | PRs | Accepted |", "|---|---|---|---|---|---|"]
    for x in b:
        claims = ", ".join(f"@{c['by']} {c['at'][5:10]}" for c in x["claims"]) or "none"
        subs = ", ".join(f"#{s['pr']} @{s['by']} {s['state']}" + (f" ({s['checks']})" if s["state"] == "open" else "")
                         for s in x["prs"]) or "none"
        acc = f"#{x['pr']} @{x['paidTo']}, {x['month']}" if x["paidTo"] else x["state"]
        md.append(f"| #{x['issue']} | {x['device']} | {x['amount']} | {claims} | {subs} | {acc} |")
    if months:
        md += ["", "| Month | Accepted | Owed |", "|---|---|---|"]
        md += [f"| {m} | {months[m]['accepted']} | ${months[m]['owed']} |" for m in sorted(months)]
    open(os.path.join(OUT, "ledger.md"), "w").write("\n".join(md) + "\n")
    print(json.dumps({k: ops["bounty"][k] for k in ("month", "owed", "accepted", "open", "claims", "prsOpen")}))


if __name__ == "__main__":
    main()
