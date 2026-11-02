import type { Metadata } from "next";
import localFont from "next/font/local";
import PostHog from "@/components/PostHog";
import { stats } from "@/lib/registry";
import "./globals.css";

const mono = localFont({ src: [{ path: "../assets/mono-400.ttf", weight: "400" }, { path: "../assets/mono-500.ttf", weight: "500" }], variable: "--font-mono" });
// Geist 1.9.0 from vercel/geist-font; license in assets/geist-OFL.txt.
const sans = localFont({ src: "../assets/geist-sans.woff2", variable: "--font-sans", weight: "100 900" });
const SITE = "https://local.sybilsolutions.ai";

export const metadata: Metadata = {
  metadataBase: new URL(SITE),
  title: { default: "Sybil AI: the model to run on your GPU or CPU", template: "%s · Sybil AI" },
  description: `Tested recipes for ${stats.gpus} hardware types: which model to run, how fast it goes, and the exact command. Tested recipes passed six checks on the card.`,
  openGraph: { siteName: "Sybil AI", type: "website" },
  twitter: { card: "summary_large_image" },
};

/** The page shell: fonts and analytics. The landing page (/) draws its own header; the registry pages share theirs in (registry)/layout.tsx. */
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${mono.variable} ${sans.variable}`}>
      <body>
        {children}
        <PostHog />
        {/* Cloudflare Web Analytics: page views and Core Web Vitals, no cookies */}
        <script defer src="https://static.cloudflareinsights.com/beacon.min.js" data-cf-beacon='{"token": "9e30875864824920804a1f95dd926025"}' />
      </body>
    </html>
  );
}
