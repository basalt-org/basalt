import { getAllPackages, Package } from "@/lib/docs";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./ui/accordion";
import Link from "next/link";

const packages = getAllPackages();

function findPackagePath(name: string): string {
  for (const pkg of packages) {
    if (pkg.package.name === name) {
      return "/docs" + pkg.path;
    }
  }
  return "";
}

export default function Sidebar({ pkg }: { pkg: Package }) {
  return (
    <aside className="w-1/5 absolute left-0 top-16">
      <Accordion type="multiple">
        {pkg.packages && pkg.packages.length > 0 && (
          <AccordionItem value="Packages" className="w-full px-4">
            <AccordionTrigger>Packages</AccordionTrigger>
            <AccordionContent>
              {pkg.packages.map((p) => (
                <div key={p.name} className="p-1">
                  <Link
                    key={p.name}
                    className="pl-4 capitalize"
                    href={findPackagePath(p.name)}
                  >
                    {p.name}
                  </Link>
                </div>
              ))}
            </AccordionContent>
          </AccordionItem>
        )}
        {pkg.modules && pkg.modules.length > 0 && (
          <AccordionItem value="Modules" className="w-full px-4">
            <AccordionTrigger>Modules</AccordionTrigger>
            <AccordionContent>
              {pkg.modules.map((p) => (
                <div className="p-1" key={p.name}>
                  <Link className="pl-4 capitalize" href={"#" + p.name}>
                    {p.name}
                  </Link>
                </div>
              ))}
            </AccordionContent>
          </AccordionItem>
        )}
      </Accordion>
    </aside>
  );
}
