import { loadDocs } from "@/lib/docs";
import { Documentation } from "@/types/mojo";
import { MainNavItem, SidebarNavItem } from "@/types/nav";

export interface DocsConfig {
  mainNav: MainNavItem[];
  sidebarNav: SidebarNavItem[];
}

export async function generateDocsConfig(): Promise<DocsConfig> {
  const docs: Documentation = await loadDocs();

  const mainNav: MainNavItem[] = [
    {
      title: "Documentation",
      href: "/docs",
    },
  ];

  const sidebarNav: SidebarNavItem[] = docs.decl.modules.map((module) => ({
    title: module.name,
    items: [
      {
        title: "Overview",
        href: `/docs/${module.name.toLowerCase()}`,
        items: [],
      },
      ...module.functions.map((fn) => ({
        title: fn.name,
        href: `/docs/${module.name.toLowerCase()}/${fn.name.toLowerCase()}`,
        items: [],
      })),
      ...module.structs.map((struct) => ({
        title: struct.name,
        href: `/docs/${module.name.toLowerCase()}/${struct.name.toLowerCase()}`,
        items: [],
      })),
    ],
  }));

  return {
    mainNav,
    sidebarNav,
  };
}
