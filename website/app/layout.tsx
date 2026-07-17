import type { Metadata } from "next";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import "./styles.css";

const buildSha = process.env.NEXT_PUBLIC_BUILD_SHA || "local";

export const metadata: Metadata = {
  title: "Credit Scoring Replication",
  description:
    "Website giới thiệu đề tài partial replication và đánh giá lại mô hình credit scoring.",
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
            <Link className="brand" href="/">
              Credit Scoring Replication
            </Link>
            <Nav />
          </div>
        </header>
        <main>{children}</main>
        <footer className="site-footer">
          <p>
            P1A: website local static-first. Chưa công bố kết quả thực nghiệm,
            chưa triển khai production.
          </p>
          <p className="build-version">Build: {buildSha}</p>
        </footer>
      </body>
    </html>
  );
}
