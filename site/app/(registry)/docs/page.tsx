import PageHead from "@/components/PageHead";
import { TERMINAL } from "@/lib/landing/pictures";

const SITE = "https://local.sybilsolutions.ai";

export const metadata = { title: "How it works", description: "How a recipe is made, checked and run; the file tree; the API and SDK." };

const TREE = `registry/                     the source of truth, tiny
  cards/<vendor>/<card>.json        a GPU: name, memory, how to detect it
  engines/<profile>.json            an engine: pinned image, arguments, config
  recipes/<vendor>/<card>/<model>.<engine>.<ctx>k.json
                                    one recipe (~400 bytes): weights, engine, settings, proof
  models.json                       each model: family, release date, what it is for
lab/            make recipes: lab.py (try, convert, render, check), catalog.py
dist/           catalog.json: everything above, rendered, 3 picks per GPU
site/           this website and its API
sdk/            js/ and python/: pick a recipe for a GPU, print the command
app/            local-ai: run a recipe on any Linux
images/         the container images we build (gateway, tabbyapi-exl3, ...)
plugin/         what the Omarchy Local AI plugin reads
data/           everything the registry held before; nothing reads it`;

const RECIPE = `{"model":"qwen3.5-9b",
 "weights":"TheMelonGod/Qwen3.5-9B-exl3@22ef1303062e0f6d0b282440f8c1f685947f4938",
 "engine":"tabbyapi-exl3@0f83e6198dc3",
 "set":{"ctx":65536,"draft":"mtp"},
 "card":"rtx-3070-ti-8gb",
 "proof":[{"at":"2026-09-25","on":"vast","gpu":"RTX 3070 Ti",
           "gates":"load chat reasoning tools context speed","tps":105.3,"prefill":1485}]}`;

export default function Docs() {
  return (
    <main>
      <PageHead title="How it works" sub="The lab rents the card, runs six checks, and writes the recipe only if all of them pass." art={TERMINAL} />

      <section><span className="label">A recipe</span><pre className="code">{RECIPE}</pre>
        <dl className="kv" style={{ marginTop: 18 }}>
          <dt>weights</dt><dd>A Hugging Face repo at an exact commit.</dd>
          <dt>engine</dt><dd>A profile in <code>registry/engines/</code>, pinned to its container image digest.</dd>
          <dt>set</dt><dd>Only the settings that differ from the profile&apos;s defaults.</dd>
          <dt>proof</dt><dd>The runs it passed: where, on what GPU, which checks, and the speed measured. <b className="warm">Sibling</b> means it ran on the same chip family because nobody rents this card; <b className="dim">earlier check</b> means it passed the older acceptance and is waiting for a full run.</dd>
        </dl>
      </section>

      <section><span className="label">The six checks</span>
        <table className="t"><tbody>
          <tr><td>load</td><td className="dim">The server lists the model within an hour</td></tr>
          <tr><td>chat</td><td className="dim">A plain question gets an answer that ends on its own</td></tr>
          <tr><td>reasoning</td><td className="dim">The thinking comes back separately, and 17 × 23 is answered 391</td></tr>
          <tr><td>tools</td><td className="dim">A weather tool is called with the right city, and its result is used in the reply</td></tr>
          <tr><td>context</td><td className="dim">A code planted in a prompt filling 85% of the window is recalled</td></tr>
          <tr><td>speed</td><td className="dim">Decode over the first 30 seconds of an answer is at least 15 tok/s. No answer is ever cut short.</td></tr>
        </tbody></table>
      </section>

      <section><span className="label">The repository</span><pre className="code tree">{TREE}</pre></section>

      <section id="api"><span className="label">API</span>
        <table className="t"><tbody>
          <tr><td><code>GET /api/v2/catalog.json</code></td><td className="dim">Everything: every GPU, its picks, every recipe with its rendered launch</td></tr>
          <tr><td><code>GET /api/v2/gpus/&lt;gpu&gt;.json</code></td><td className="dim">One GPU and its top recipes, launches included</td></tr>
          <tr><td><code>GET /api/v2/pick?gpu=RTX%203090&amp;vram=24</code></td><td className="dim">The recommended recipe for the GPU a program detected</td></tr>
        </tbody></table>
        <pre className="code" style={{ marginTop: 14 }}>{`curl -s "${SITE}/api/v2/pick?gpu=RTX%204090&vram=24" | jq .recipe.launch.image`}</pre>
      </section>

      <section id="sdk"><span className="label">SDK</span>
        <pre className="code">{`// JavaScript (sdk/js): no dependencies
import { pick, command } from "@local-ai/registry";
const r = await pick({ gpu: "NVIDIA GeForce RTX 4090", vram: 24 });
console.log(command(r));

# Python (sdk/python): standard library only
from local_ai_registry import pick, command
print(command(pick(gpu="NVIDIA GeForce RTX 4090", vram=24)))`}</pre>
      </section>
    </main>
  );
}
