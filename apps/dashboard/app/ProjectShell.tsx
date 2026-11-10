"use client";

import type { ReactNode } from "react";
import Link from "next/link";

const projects = [
  { href: "/", label: "总览" },
  { href: "/a-share", label: "A股" },
  { href: "/tqqq", label: "TQQQ" },
  { href: "/us-overnight", label: "美股隔夜" },
];

export default function ProjectShell({ active, eyebrow, title, description, children }:{active:string;eyebrow:string;title:string;description:string;children:ReactNode}) {
  return <div className="project-page">
    <header className="project-header"><Link className="project-logo" href="/"><span>M</span><b>MoneyMore</b></Link><nav>{projects.map(project=><Link key={project.href} className={active===project.href?"active":""} href={project.href}>{project.label}</Link>)}</nav></header>
    <main className="project-main"><section className="project-intro"><small>{eyebrow}</small><h1>{title}</h1><p>{description}</p></section>{children}</main>
  </div>;
}
