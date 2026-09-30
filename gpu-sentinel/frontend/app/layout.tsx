import "./globals.css";
import type { Metadata } from "next";
import Shell from "@/components/Shell";

export const metadata: Metadata = {
  title: "GPU Sentinel AI",
  description: "Data-center performance & health intelligence for GPU clusters",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark">
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
