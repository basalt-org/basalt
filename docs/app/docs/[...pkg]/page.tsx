import ModuleCard from "@/components/module-card";
import Sidebar from "@/components/sidebar";
import {
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { findPackage } from "@/lib/docs";
import React, { Fragment, useMemo } from "react";

export default function Package({ params }: { params: { pkg: string[] } }) {
  const Crumbs = useMemo(() => {
    const generateHref = (index: number) => {
      return `/docs/${params.pkg.slice(0, index + 1).join("/")}`;
    };

    return (
      <Breadcrumb className="text-primary font-semibold p-4">
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink href="/">Home</BreadcrumbLink>
          </BreadcrumbItem>
          {params.pkg.map((pkg, index) => (
            <Fragment key={pkg}>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbLink
                  href={generateHref(index)}
                  className="capitalize"
                >
                  {pkg}
                </BreadcrumbLink>
              </BreadcrumbItem>
            </Fragment>
          ))}
        </BreadcrumbList>
      </Breadcrumb>
    );
  }, [params.pkg]);

  const pkg = findPackage(params.pkg);

  if (!pkg) {
    return <div>Package not found</div>;
  }

  const modules = pkg.modules || [];

  return (
    <main>
      <div className="flex items-center h-full">
        <Sidebar pkg={pkg} />
        <div className="w-4/5 p-4 grid grid-cols-1 gap-4 translate-x-1/4">
          {modules.length > 0 &&
            modules.map((mdl) => <ModuleCard mdl={mdl} key={mdl.name} />)}
        </div>
      </div>
      <div className="fixed bottom-0 right-0">{Crumbs}</div>
    </main>
  );
}
