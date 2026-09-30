"use client";
// Old URL: redirect to Ask Sentinel.
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function OldCopilot() {
  const router = useRouter();
  useEffect(() => { router.replace("/ask/"); }, [router]);
  return null;
}
