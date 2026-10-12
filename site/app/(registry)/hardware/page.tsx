import HardwareDirectory from "@/components/HardwareDirectory";
import PageHead from "@/components/PageHead";
import { GPU } from "@/lib/landing/pictures";
import { hardwareDirectoryRows } from "@/lib/configurations";
import { bare, stats } from "@/lib/registry";

export const metadata = { title: "Hardware", description: "Search and sort local AI hardware, original manufacturer photos, recorded model configurations, speeds and launch commands." };

export default function Hardware() {
  return (
    <main>
      <PageHead title="Hardware" sub="Original hardware photos and specifications. Open any entry for every recorded model configuration and its exact command." art={GPU}>
        <div className="facts"><span><b>{stats.gpus}</b> with recipes</span><span><b>{stats.recipes}</b> recipes</span><span><b>{stats.hardware}</b> hardware entries</span></div>
      </PageHead>
      <section id="gpus"><HardwareDirectory rows={hardwareDirectoryRows(bare)} /></section>
    </main>
  );
}
