import Sidebar from "@/components/sidebar";
import { Breadcrumb, BreadcrumbList, BreadcrumbItem, BreadcrumbLink, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
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
                <BreadcrumbLink href={generateHref(index)} className="capitalize">{pkg}</BreadcrumbLink>
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

  return (
    <main>
      <Sidebar pkg={pkg} />
    </main>
  );
}
