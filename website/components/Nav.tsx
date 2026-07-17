"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Trang chủ" },
  { href: "/gioi-thieu/", label: "Giới thiệu" },
  { href: "/datasets/", label: "Dữ liệu" },
  { href: "/phuong-phap/", label: "Phương pháp" },
  { href: "/tien-do/", label: "Tiến độ" },
  { href: "/ket-qua/", label: "Kết quả" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <nav className="site-nav" aria-label="Điều hướng chính">
      {navItems.map((item) => {
        const isActive =
          item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);

        return (
          <Link
            aria-current={isActive ? "page" : undefined}
            className={isActive ? "active" : undefined}
            href={item.href}
            key={item.href}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
