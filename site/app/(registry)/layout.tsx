import Seen from "@/components/Seen";
import { SiteFoot, SiteNav } from "@/components/SiteNav";
import s from "../landing.module.css";

// The registry pages wear the landing's header, footer and charcoal; their content sits in the narrower .wrap.
export default function RegistryLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className={s.page}>
      <SiteNav />
      <div className="wrap">{children}</div>
      <SiteFoot />
      <Seen />
    </div>
  );
}
