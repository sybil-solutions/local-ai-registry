"use client";
import s from "@/app/landing.module.css";
import { track } from "./PostHog";

// The button that downloads the desktop app. A plain link, so it works with scripts off; the click is counted when
// analytics are on. Its address is one of the redirects in public/_redirects, which lead to the newest release.
export default function Download({ href, os, from, children }: { href: string; os: string; from: string; children: React.ReactNode }) {
  return <a href={href} className={`${s.btn} ${s.dl}`} onClick={() => track("app_download_clicked", { os, from })}>{children}</a>;
}
