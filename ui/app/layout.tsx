import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SEED — Local Kabbalistic Cognition Prototype",
  description: "A guided, inspectable local proof of concept for Form, Flow, Accord, and the Tree-of-Life cognitive graph.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
