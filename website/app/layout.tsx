import type { Metadata, Viewport } from "next";
import Image from "next/image";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import "./styles.css";

const buildSha = process.env.NEXT_PUBLIC_BUILD_SHA || "local";

export const metadata: Metadata = {
  metadataBase: new URL("http://34.142.206.15"),
  title: "Tái lập và đánh giá lại mô hình tính điểm tín dụng",
  description:
    "Website báo cáo tiến độ và công bố kết quả kiểm chứng của đề tài tính điểm tín dụng.",
  manifest: "/site.webmanifest",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "16x16 32x32 48x48" },
      { url: "/brand/csr-favicon.png", sizes: "973x973", type: "image/png" },
      { url: "/brand/csr-icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/brand/csr-icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    apple: [{ url: "/brand/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  openGraph: {
    title: "Tái lập và đánh giá lại mô hình tính điểm tín dụng",
    description:
      "Website báo cáo tiến độ và công bố kết quả kiểm chứng của đề tài tính điểm tín dụng.",
    images: [{ url: "/brand/csr-logo-full.png", alt: "Logo dự án Tái lập và đánh giá lại mô hình tính điểm tín dụng" }],
  },
};

export const viewport: Viewport = {
  themeColor: "#E30613",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi">
      <body>
        <header className="site-header">
          <div className="header-inner">
            <Link
              aria-label="Về trang chủ – CSR Credit Scoring Replication"
              className="brand"
              href="/"
            >
              <Image
                alt=""
                className="brand-logo"
                height="521"
                priority
                sizes="(max-width: 560px) 190px, 240px"
                src="/brand/csr-logo-header.png"
                unoptimized
                width="2083"
              />
            </Link>
            <Nav />
          </div>
        </header>
        <main>{children}</main>
        <footer className="site-footer">
          <p className="build-version">Build: {buildSha}</p>
        </footer>
      </body>
    </html>
  );
}
