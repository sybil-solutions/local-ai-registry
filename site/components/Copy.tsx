"use client";
import { useEffect, useState } from "react";
import { track } from "./PostHog";

export default function Copy({ text, event, props }: { text: string; event: string; props?: Record<string, unknown> }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  useEffect(() => { setState("idle"); }, [text]);
  useEffect(() => {
    if (state === "idle") return;
    const timeout = setTimeout(() => setState("idle"), 1500);
    return () => clearTimeout(timeout);
  }, [state]);
  return (
    <button className="btn copy" onClick={async () => {
      try {
        await navigator.clipboard.writeText(text);
        setState("copied");
        track(event, props);
      } catch { setState("failed"); }
    }}>
      {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy"}
    </button>
  );
}
