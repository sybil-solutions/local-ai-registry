import ModelDirectory from "@/components/ModelDirectory";
import PageHead from "@/components/PageHead";
import { NET } from "@/lib/landing/pictures";
import { modelDirectoryRows } from "@/lib/configurations";

export const metadata = { title: "Models", description: "Search and compare local AI models, rendered Hugging Face model cards, hardware configurations, measured speeds and exact launch commands." };

export default function Models() {
  return (
    <main>
      <PageHead title="Models" sub="Every model with a recorded recipe. Open a model for its Hugging Face card, hardware configurations, measured speeds and exact commands." art={NET} />
      <section><ModelDirectory rows={modelDirectoryRows()} /></section>
    </main>
  );
}
