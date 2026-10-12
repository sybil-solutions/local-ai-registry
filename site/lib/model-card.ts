// Reads the prefetched Hugging Face model cards (scripts/model-cards.mjs -> model-cards.data.json)
// and turns the raw README Markdown into sanitized HTML for the server-rendered <ModelCard>.
//
// Security model: the README is untrusted. We parse GFM Markdown, keep inline HTML, then
//  1. rewrite relative links/images against the real source repo,
//  2. allow <img> only from trusted Hugging Face / GitHub-raw hosts (others become plain links),
//  3. drop any link/image to a private or local-network host,
//  4. sanitize the tree with a strict allow-list — scripts, iframes, forms, event handlers and
//     javascript: URLs cannot survive.
// No remote content is fetched at render time; images load in the visitor's browser only from
// the trusted hosts above. There is no client-side iframe embedding.
import { unified, type Plugin } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeRaw from "rehype-raw";
import rehypeSlug from "rehype-slug";
import rehypeSanitize, { defaultSchema, type Options as SanitizeSchema } from "rehype-sanitize";
import rehypeStringify from "rehype-stringify";
import { visit } from "unist-util-visit";
import type { Root, Element } from "hast";
import data from "./model-cards.data.json";

export type CardStatus = "ok" | "gated" | "not-found" | "no-card" | "no-source" | "error";
export type ModelCardData = {
  modelId: string;
  name: string;
  repo?: string | null;
  repoUrl?: string;
  revision?: string;
  resolvedVia?: string | null;
  gated?: boolean;
  status: CardStatus;
  note?: string;
  fetchedAt?: string;
  markdown?: string;
};

// The generated JSON widens the status field to string; assert our own shape once here.
const cardData = data as unknown as { cards: Record<string, ModelCardData> };
const cards = cardData.cards;

export const getModelCard = (modelId: string): ModelCardData | undefined => cards[modelId];
export const hasModelCard = (modelId: string): boolean => cards[modelId]?.status === "ok";

// Hosts whose <img> we let the browser load. Everything else is downgraded to a link so that
// the card can never make the visitor fetch an arbitrary or local-network image.
const TRUSTED_IMG_HOSTS = ["huggingface.co", "hf.co", "raw.githubusercontent.com"];

function isPrivateHost(host: string): boolean {
  const h = host.replace(/^\[|\]$/g, "").toLowerCase();
  if (h === "localhost" || h.endsWith(".local") || h.endsWith(".internal") || !h.includes(".")) return true;
  if (/^127\.|^10\.|^192\.168\.|^169\.254\.|^0\./.test(h)) return true;
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(h)) return true;
  if (h === "::1" || h.startsWith("fe80:") || h.startsWith("fc") || h.startsWith("fd")) return true;
  return false;
}

// Resolve a README URL against the source repo; returns an absolute https URL, an in-page
// anchor, or null if it is unusable / points at a private host. `raw` picks the file-content
// base (for images) vs. the human blob page (for links).
function resolveUrl(url: string | undefined, repo: string | undefined, rev: string, raw: boolean): string | null {
  if (!url) return null;
  const u = url.trim();
  if (!u) return null;
  if (u.startsWith("#")) return raw ? null : `#${defaultSchema.clobberPrefix ?? "user-content-"}${u.slice(1)}`;
  if (u.startsWith("data:")) return raw ? u : null; // inline data: only as image, never as link target
  let abs: URL;
  try {
    if (/^[a-z][a-z0-9+.-]*:/i.test(u)) abs = new URL(u);
    else if (u.startsWith("//")) abs = new URL("https:" + u);
    else if (repo) abs = new URL(u.replace(/^\.\//, ""), `https://huggingface.co/${repo}/${raw ? "resolve" : "blob"}/${rev}/`);
    else return null;
  } catch {
    return null;
  }
  if (abs.protocol !== "https:" && abs.protocol !== "http:") return null;
  if (isPrivateHost(abs.hostname)) return null;
  return abs.toString();
}

// rehype transform: fix up link/image URLs against the repo and quarantine untrusted images.
const rehypeRepoUrls: Plugin<[string | undefined, string], Root> = (repo, rev) => (tree) => {
  visit(tree, "element", (node: Element, index, parent) => {
      if (node.tagName === "a" && node.properties.href != null) {
        const href = resolveUrl(String(node.properties.href), repo, rev, false);
        if (href) {
          node.properties.href = href;
          if (href.startsWith("#")) {
            delete node.properties.rel;
            delete node.properties.target;
          } else {
            node.properties.rel = ["nofollow", "noopener", "noreferrer"];
            node.properties.target = "_blank";
          }
        } else delete node.properties.href;
        return;
      }
      if (node.tagName !== "img") return;
      const src = resolveUrl(node.properties.src != null ? String(node.properties.src) : undefined, repo, rev, true);
      let host = "";
      try { host = src && src.startsWith("http") ? new URL(src).hostname : ""; } catch { host = ""; }
      const trusted = !!src && (src.startsWith("data:") || TRUSTED_IMG_HOSTS.some((h) => host === h || host.endsWith("." + h)));
      if (src && trusted) {
        node.properties.src = src;
        node.properties.loading = "lazy";
        node.properties.referrerPolicy = "no-referrer";
      } else if (src && parent && index != null) {
        // Untrusted image host: replace the <img> with an explicit, non-auto-fetched link.
        const alt = String(node.properties.alt || "image");
        parent.children[index] = {
          type: "element", tagName: "a",
          properties: { href: src, rel: ["nofollow", "noopener", "noreferrer"], target: "_blank", className: ["mc-extimg"] },
          children: [{ type: "text", value: `Image: ${alt}` }],
        };
      } else if (parent && index != null) {
        parent.children[index] = { type: "text", value: "" };
      }
    });
};

// Strict allow-list: defaultSchema already forbids script/style/iframe/form/object/embed and
// every on* handler, and only permits safe URL protocols. We add the table/figure/detail tags
// and the few attributes our rewrite step sets.
const schema: SanitizeSchema = {
  ...defaultSchema,
  tagNames: Array.from(new Set([
    ...(defaultSchema.tagNames ?? []),
    "details", "summary", "figure", "figcaption", "picture", "source", "abbr", "kbd", "samp", "mark",
  ])),
  attributes: {
    ...defaultSchema.attributes,
    img: [...(defaultSchema.attributes?.img ?? []), "loading", "referrerPolicy", "width", "height", "align"],
    // keep aria/href from the default, add rel/target and the class our untrusted-image downgrade sets
    a: ["ariaDescribedBy", "ariaLabel", "ariaLabelledBy", "dataFootnoteBackref", "dataFootnoteRef", "href", "rel", "target", ["className", "mc-extimg", "data-footnote-backref"]],
    "*": [...(defaultSchema.attributes?.["*"] ?? []), "align"],
  },
  protocols: {
    ...defaultSchema.protocols,
    src: ["http", "https", "data"],
    href: ["http", "https", "mailto"],
  },
};

const htmlCache = new Map<string, string>();

/** Render a cached card's Markdown to sanitized HTML. Memoized per model for the static build. */
export async function renderCardHtml(card: ModelCardData): Promise<string> {
  if (!card.markdown) return "";
  const cached = htmlCache.get(card.modelId);
  if (cached != null) return cached;
  // Drop the leading YAML frontmatter block before rendering the body.
  const body = card.markdown.replace(/^\uFEFF?\s*---\r?\n[\s\S]*?\r?\n---\r?\n/, "");
  const file = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSlug)
    .use(rehypeRepoUrls, card.repo ?? undefined, card.revision || "main")
    .use(rehypeSanitize, schema)
    .use(rehypeStringify)
    .process(body);
  const html = String(file);
  htmlCache.set(card.modelId, html);
  return html;
}
