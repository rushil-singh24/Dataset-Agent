import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DataChat AI",
  description: "Local conversational analysis for uploaded datasets"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
