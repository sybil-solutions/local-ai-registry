import s from "./RegistryCards.module.css";

const LOGOS: Record<string, string> = { nvidia: "/logos/nvidia.svg", amd: "/logos/amd.svg", intel: "/logos/intel.svg", apple: "/logos/apple.svg" };

export default function VendorLogo({ vendor }: { vendor: string }) {
  const src = LOGOS[vendor];
  // Generic CPU entries do not identify a manufacturer; don't imply one.
  return src ? <img className={s.vendorLogo} src={src} width={48} height={32} alt="" /> : <span className="chip">{vendor.toUpperCase()}</span>;
}
