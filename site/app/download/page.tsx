import type { Metadata } from "next";
import Copy from "@/components/Copy";
import Download from "@/components/Download";
import { RELEASES, SiteFoot, SiteNav } from "@/components/SiteNav";
import copy from "@/lib/landing/copy.json";
import s from "../landing.module.css";

// Where the desktop app is downloaded. The Mac button and the Linux line go through /download/mac and /install.sh,
// which public/_redirects sends to the newest release, so this page names no version and needs no change for a new one.
// The words are in lib/landing/copy.json (d…).
export const metadata: Metadata = { title: copy.dHead, description: copy.dLede };
const INSTALL = "curl -fsSL https://local.sybilsolutions.ai/install.sh | bash";
// Whether the newest release holds the Mac app (Local-AI-mac-arm64.dmg). It has since 0.7.10. A release published
// with the Linux files alone (scripts/publish.sh --linux in the app's repo) has none: the button would then lead
// nowhere, so set this to false until the Mac app is added to that release.
const MAC = true;

export default function DownloadPage() {
  return (
    <div className={s.page}>
      <SiteNav />
      <section className={`${s.section} ${s.top}`}>
        <div className={s.head}><span className={s.label}>{copy.dHead}</span><a href={`${RELEASES}/releases`} className={s.more}>{copy.dAll}</a></div>
        <p className={s.lede}>{copy.dLede}</p>
        <div className={s.gets}>
          {MAC && (
            <div className={s.get}>
              <h3>{copy.dMac}</h3>
              <p>{copy.dMacNote}</p>
              <Download href="/download/mac" os="mac" from="download">{copy.dMacBtn}</Download>
            </div>
          )}
          <div className={s.get}>
            <h3>{copy.dLinux}</h3>
            <p>{copy.dLinuxNote}</p>
            <div className={s.cmd}>
              <span className={s.prompt}>$</span>
              <code>{INSTALL}</code>
              <Copy text={INSTALL} event="app_install_copied" props={{ from: "download" }} />
            </div>
          </div>
        </div>
      </section>
      <SiteFoot />
    </div>
  );
}
