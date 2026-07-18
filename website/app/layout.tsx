import type { Metadata } from "next";
import Link from "next/link";
import { Nav } from "@/components/Nav";
import "./styles.css";

const buildSha = process.env.NEXT_PUBLIC_BUILD_SHA || "local";

export const metadata: Metadata = {
  title: "Tái lập và đánh giá lại mô hình tính điểm tín dụng",
  description:
    "Website báo cáo tiến độ và công bố kết quả kiểm chứng của đề tài tính điểm tín dụng.",
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
              Tái lập và đánh giá lại mô hình tính điểm tín dụng
            </Link>
            <Nav />
          </div>
        </header>
        <main>{children}</main>
        <footer className="site-footer">
          <p>
            Production: http://34.142.206.15. Phase 1 còn pending kiểm thử
            rollback; chưa công bố kết quả thực nghiệm.
          </p>
          <p className="build-version">Build: {buildSha}</p>
        </footer>
      </body>
    </html>
  );
}
