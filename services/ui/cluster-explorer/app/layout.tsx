import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import Link from "next/link";
import { LayoutGrid, MessageSquare, Search } from "lucide-react";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ZTI Cluster Explorer | Zendesk Ticket Intelligence",
  description: "Explore and analyze ticket clusters with AI-powered insights",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={inter.variable}>
      <body className={inter.className}>
        <ThemeProvider defaultTheme="dark">
          {/* Header following txt2kg pattern */}
          <header className="border-b border-border/50 backdrop-blur-md dark:bg-background/95 bg-background sticky top-0 z-50 shadow-sm">
            <div className="container mx-auto px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-lg bg-[#76b900] flex items-center justify-center">
                  <LayoutGrid className="h-5 w-5 text-white" />
                </div>
                <div>
                  <span className="text-xl font-bold gradient-text">ZTI Cluster Explorer</span>
                </div>
              </div>
              <nav className="flex items-center gap-4">
                <Link
                  href="/"
                  className="flex items-center gap-2 text-sm font-medium rounded-lg px-3 py-2 transition-colors hover:bg-muted"
                >
                  <LayoutGrid className="h-4 w-4" />
                  <span>Clusters</span>
                </Link>
                <Link
                  href="/search"
                  className="flex items-center gap-2 text-sm font-medium rounded-lg px-3 py-2 transition-colors hover:bg-muted"
                >
                  <Search className="h-4 w-4" />
                  <span>Search</span>
                </Link>
                <Link
                  href="/chat"
                  className="flex items-center gap-2 text-sm font-medium rounded-lg px-3 py-2 transition-colors border border-[#76b900]/40 text-[#76b900] bg-[#76b900]/10 hover:bg-[#76b900]/20"
                >
                  <MessageSquare className="h-4 w-4" />
                  <span>Tier-0 Chat</span>
                </Link>
                <ThemeToggle />
              </nav>
            </div>
          </header>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
