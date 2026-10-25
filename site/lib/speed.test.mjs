// node --experimental-strip-types --test site/lib/speed.test.mjs   (make check; CI)
import { test } from "node:test";
import assert from "node:assert/strict";
import { isEstimated, tpsValue, tpsPhrase, fastestMeasured, topSpeed } from "./speed.ts";

const measured = (tps) => ({ tps });
const estimate = (tps) => ({ tps, tps_estimated: true });

test("only the typed flag marks an estimate, never a note's wording", () => {
  assert.equal(isEstimated(estimate(36)), true);
  assert.equal(isEstimated(measured(36)), false);
  assert.equal(isEstimated({ tps: 36, note: "an estimate of throughput, reported by its publisher" }), false);
  assert.equal(isEstimated({ tps: 36, tps_estimated: false }), false);
});

test("an estimate is shown as one wherever a number is printed", () => {
  assert.equal(tpsValue(estimate(35.9)), "≈36");
  assert.equal(tpsValue(measured(35.9)), "36");
  assert.equal(tpsValue({ tps: null }), "–");
  assert.equal(tpsPhrase(estimate(35.9)), "≈36 tok/s (estimate)");
  assert.equal(tpsPhrase(measured(105.3)), "105 tok/s");
});

test("Fastest compares measured speeds only", () => {
  const rs = [{ key: "a", p: measured(20) }, { key: "b", p: estimate(36) }, { key: "c", p: measured(25) }];
  assert.equal(fastestMeasured(rs, (r) => r.p).key, "c");  // the higher estimate does not win
  assert.equal(fastestMeasured([rs[1]], (r) => r.p), null);  // an estimate alone is never "Fastest"
  assert.equal(fastestMeasured([{ key: "n", p: { tps: null } }], (r) => r.p), null);
  assert.equal(fastestMeasured([{ key: "x", p: measured(20) }, { key: "y", p: measured(20) }], (r) => r.p).key, "x");  // first on a tie
});

test("the highest speed is a measured one when any exists, else a flagged estimate", () => {
  assert.deepEqual(topSpeed([measured(20), estimate(36)]), { tps: 20, estimated: false });
  assert.deepEqual(topSpeed([estimate(30), estimate(36)]), { tps: 36, estimated: true });
  assert.deepEqual(topSpeed([{ tps: null }]), { tps: null, estimated: false });
  assert.deepEqual(topSpeed([]), { tps: null, estimated: false });
});
