"use client";
import { PageHead } from "@/components/ui";
import { AskSentinel } from "@/components/AskSentinel";

export default function Ask() {
  return (
    <>
      <PageHead title="Ask Sentinel" desc="Ask about your cluster in plain language. Answers come from GPU Sentinel's own analysis of your live telemetry and incidents, running on this server. Nothing is sent to an outside service." />
      <AskSentinel />
    </>
  );
}
