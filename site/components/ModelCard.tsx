// The model's real Hugging Face card: prefetched README Markdown rendered to sanitized HTML,
// with a provenance line naming the exact source repository and how it was retrieved. When a
// card is genuinely unavailable (gated, missing, or no public repo) it says so honestly and
// links out instead of inventing a body. Rendered as an async server component at build time.
import { getModelCard, renderCardHtml, type CardStatus } from "@/lib/model-card";
import styles from "./ModelCard.module.css";

const VIA_LABEL: Record<string, string> = {
  "catalog.hf": "catalog model link",
  "weight upstream": "weight upstream",
  "base_model chain": "base-model metadata",
  "Hugging Face search": "Hugging Face search",
};

const UNAVAILABLE: Record<Exclude<CardStatus, "ok">, string> = {
  gated: "This model's source repository is gated or private on Hugging Face, so its card cannot be mirrored here.",
  "not-found": "The source repository could not be found on Hugging Face.",
  "no-card": "The source repository does not publish a README model card.",
  "no-source": "No public Hugging Face repository is linked for this model, and none could be resolved from its weights.",
  error: "The model card could not be retrieved from Hugging Face.",
};

// Git commit SHAs are 40 hex chars; show the short form, otherwise the ref name (e.g. "main").
const shortRev = (rev?: string) => (rev && /^[0-9a-f]{40}$/i.test(rev) ? rev.slice(0, 7) : rev || "main");

export default async function ModelCard({ modelId }: { modelId: string }) {
  const card = getModelCard(modelId);

  if (!card || card.status !== "ok") {
    const note = card?.note ?? (card && card.status !== "ok" ? UNAVAILABLE[card.status] : UNAVAILABLE["no-source"]);
    return (
      <div className={styles.unavailable}>
        <p>{note}</p>
        {card?.repoUrl && (
          <p className="dim">
            Source: <a href={card.repoUrl} rel="nofollow noopener noreferrer" target="_blank">{card.repo} ↗</a>
          </p>
        )}
      </div>
    );
  }

  const html = await renderCardHtml(card);
  const via = card.resolvedVia ? VIA_LABEL[card.resolvedVia] ?? card.resolvedVia : null;
  return (
    <div>
      <p className={styles.meta}>
        Source{" "}
        <a href={card.repoUrl} rel="nofollow noopener noreferrer" target="_blank">{card.repo} ↗</a>
        {" · "}pinned {shortRev(card.revision)}
        {via && <>{" · via "}{via}</>}
        {card.gated && <>{" · gated upstream"}</>}
        {card.fetchedAt && <>{" · retrieved "}{new Date(card.fetchedAt).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })}</>}
      </p>
      <article className={styles.card} dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
