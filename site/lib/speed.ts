// Decode speeds as the lab recorded them. A proof whose tps the lab could only estimate (characters/4: the server had no
// tokenize route) carries `tps_estimated: true` (lab/lab.py gates). Nothing here imports the catalog, so
// lib/speed.test.mjs runs it under plain node.
export type Speed = { tps: number | null; tps_estimated?: boolean };

export const isEstimated = (p: Speed) => p.tps_estimated === true;

/** "36", "≈36" for an estimate, "–" when nothing was recorded. */
export const tpsValue = (p: Speed) => (p.tps == null ? "–" : `${isEstimated(p) ? "≈" : ""}${Math.round(p.tps)}`);

/** For prose such as a page description: "36 tok/s", "≈36 tok/s (estimate)". */
export const tpsPhrase = (p: Speed) => (p.tps == null ? "no recorded decode speed" : `${tpsValue(p)} tok/s${isEstimated(p) ? " (estimate)" : ""}`);

/** The item with the highest measured decode speed. Estimates never take part, so this is null when nothing was measured. */
export function fastestMeasured<T>(items: T[], speed: (x: T) => Speed): T | null {
  let best: T | null = null, top = -Infinity;
  for (const x of items) {
    const p = speed(x);
    if (p.tps != null && !isEstimated(p) && p.tps > top) { best = x; top = p.tps; }
  }
  return best;
}

/** The highest decode speed of several proofs: the highest measured one, or only when none was measured the highest
 *  estimate, flagged as one. */
export function topSpeed(proofs: Speed[]): { tps: number | null; estimated: boolean } {
  const measured = proofs.filter((p) => p.tps != null && !isEstimated(p)).map((p) => p.tps as number);
  if (measured.length) return { tps: Math.max(...measured), estimated: false };
  const estimates = proofs.filter((p) => p.tps != null).map((p) => p.tps as number);
  return estimates.length ? { tps: Math.max(...estimates), estimated: true } : { tps: null, estimated: false };
}
