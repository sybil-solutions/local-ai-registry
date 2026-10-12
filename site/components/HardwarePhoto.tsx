// An original manufacturer or board-vendor photo for a hardware entry.
// Images are downloaded from the original product page / press CDN, resized to a
// web-sized asset under /public/hardware, and served locally with attribution — we
// never hotlink opaque remote images. The mapping lives in lib/hardware-photos.json.
//
// Server-compatible (no client hooks): the framed image links to its original source
// page, carries a provenance caption, and tags same-family shots as representative.
// Returns null when no original image exists for the id after research.
import s from "./HardwarePhoto.module.css";
import photos from "@/lib/hardware-photos.json";

export type HardwarePhotoMeta = {
  /** Local, self-hosted, web-sized asset path, e.g. "/hardware/rtx-4090.jpg". */
  src: string;
  /** Pixel dimensions of the stored asset. */
  width: number;
  height: number;
  /** Alt text describing the pictured hardware (the exact SKU when not representative). */
  alt: string;
  /** The original product page / press release the image came from. */
  sourceUrl: string;
  /** Who owns that page (NVIDIA, Apple, AMD, Intel, a board partner…). */
  publisher: string;
  /** One-line provenance shown beneath the image. */
  caption?: string;
  /** True when the image is of the same family/chassis rather than this exact SKU. */
  representative?: boolean;
};

const MAP = photos as Record<string, HardwarePhotoMeta>;

/** The original photo for a hardware id, or null when none exists. Safe in client code. */
export const hardwarePhoto = (id: string): HardwarePhotoMeta | null => MAP[id] ?? null;

export default function HardwarePhoto({ id, compact = false }: { id: string; compact?: boolean }) {
  const p = MAP[id];
  if (!p) return null;
  // When the shot is only representative of the family, describe what is actually
  // pictured rather than asserting the exact SKU in the alt text.
  const alt = p.representative && p.caption ? `Representative image: ${p.caption}` : p.alt;
  const title = `${p.alt} — original image, ${p.publisher}`;
  return (
    <figure className={`${s.fig}${compact ? ` ${s.compact}` : ""}`}>
      <a className={s.frame} href={p.sourceUrl} rel="nofollow noopener" target="_blank" title={title}>
        <img className={s.img} src={p.src} width={p.width} height={p.height} alt={alt} loading="lazy" decoding="async" />
        {p.representative && <span className={s.tag}>Representative</span>}
      </a>
      {!compact && (
        <figcaption className={s.cap}>
          {p.caption ? `${p.caption} ` : ""}
          <a href={p.sourceUrl} rel="nofollow noopener" target="_blank">{p.publisher} ›</a>
        </figcaption>
      )}
    </figure>
  );
}
