import { loadDocs } from "@/lib/docs";
import { Documentation } from "@/types/mojo";
import { MainNavItem, SidebarNavItem } from "@/types/nav";

export interface DocsConfig {
  mainNav: MainNavItem[];
  sidebarNav: SidebarNavItem[];
}

function capitalizeFirstLetter(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export async function generateDocsConfig(): Promise<DocsConfig> {
  const docs: Documentation = await loadDocs();

  const mainNav: MainNavItem[] = [
    {
      title: "Documentation",
      href: "/docs",
    },
  ];

  const main_directories = docs.decl.packages;
  const sidebarNav: SidebarNavItem[] = [];

  for (const main_directory of main_directories) {
    const main_directory_name = capitalizeFirstLetter(main_directory.description);
    const main_directory_modules = main_directory.modules;

    const main_directory_nav: SidebarNavItem = {
      title: main_directory_name,
      href: `/docs/${main_directory_name}`,
      items: [],
    };

    for (const module of main_directory_modules) {
      const module_name = capitalizeFirstLetter(module.name);
      const module_functions = module.functions;
      const module_structs = module.structs;

      const module_nav: SidebarNavItem = {
        title: module_name,
        href: `/docs/${main_directory_name}/${module_name}`,
        items: [],
      };

      for (const fn of module_functions) {
        const fn_name = capitalizeFirstLetter(fn.name);
        const fn_overloads = fn.overloads;

        const fn_nav: SidebarNavItem = {
          title: fn_name,
          href: `/docs/${main_directory_name}/${module_name}/${fn_name}`,
          items: [],
        };

        for (const overload of fn_overloads) {
          const overload_name = capitalizeFirstLetter(overload.name);

          const overload_nav: SidebarNavItem = {
            title: overload_name,
            href: `/docs/${main_directory_name}/${module_name}/${fn_name}/${overload_name}`,
            items: [],
          };

          fn_nav.items.push(overload_nav);
        }

        module_nav.items.push(fn_nav);
      }

      for (const struct of module_structs) {
        const struct_name = capitalizeFirstLetter(struct.name);

        const struct_nav: SidebarNavItem = {
          title: struct_name,
          href: `/docs/${main_directory_name}/${module_name}/${struct_name}`,
          items: [],
        };

        module_nav.items.push(struct_nav);
      }

      main_directory_nav.items.push(module_nav);
    }

    sidebarNav.push(main_directory_nav);
  }

  return {
    mainNav,
    sidebarNav,
  };
}
