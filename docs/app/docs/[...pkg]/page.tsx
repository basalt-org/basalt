import { Separator } from "@/components/ui/separator";
import { ArrowLeft } from "lucide-react";
import { findPackage } from "@/lib/docs";
import ModuleItem from "@/components/module_item";

export default function Package({ params }: { params: { pkg: string[] } }) {
  const CurrentPackage = findPackage(params.pkg);

  if (!CurrentPackage) {
    return (
      <main className="flex items-center justify-center pt-32">
        <div className="w-full max-w-6xl">
          <h1 className="text-3xl font-semibold text-center">
            Package not found
          </h1>
        </div>
      </main>
    );
  }

  const hasModules =
    CurrentPackage.modules && CurrentPackage.modules.length > 0;
  const hasSubPackages =
    CurrentPackage.packages && CurrentPackage.packages.length > 0; 

  return (
    <main className="flex items-center justify-center pt-8">
      <aside className="hidden md:block w-80 h-screen fixed left-0 top-16 border-r border-primary/20 overflow-y-auto dark:text-white text-gray-900">
        {params.pkg.length > 1 ? (
          <>
            <a
              href={`/docs/${params.pkg.slice(0, -1).join("/")}`}
              className="block p-2 text-primary/90 hover:text-primary"
            >
              <ArrowLeft className={"inline-block -mt-1 mr-2 w-4 h-4"} />
              Back
            </a>
            <Separator />
          </>
        ) : (
          <>
            <a
              href="/"
              className="block p-2 text-primary/90 hover:text-primary"
            >
              <ArrowLeft className={"inline-block -mt-1 mr-2 w-4 h-4"} />
              Back
            </a>
            <Separator />
          </>
        )}

        {hasModules && (
          <h2 className="text-lg font-semibold p-2 text-primary/90">Modules</h2>
        )}

        {hasModules &&
          CurrentPackage.modules!.map((mod) => (
            <div key={mod.name}>
              <a
                href={`#${mod.name}`}
                className="block p-2 pl-4 text-primary/90 hover:text-primary capitalize"
              >
                {mod.name}
              </a>
            </div>
          ))}

        {hasSubPackages && <Separator />}

        {hasSubPackages && (
          <h2 className="text-lg font-semibold p-2 text-primary/90">
            Packages
          </h2>
        )}

        {hasSubPackages &&
          CurrentPackage.packages!.map((pkg) => (
            <div key={pkg.name}>
              <a
                href={`/docs/${params.pkg.join("/")}/${pkg.name}`}
                className="block p-2 pl-4 text-primary/90 hover:text-primary capitalize"
              >
                {pkg.name}
              </a>
            </div>
          ))}
        <div className="h-16" />
      </aside>
      <div className="w-full max-w-6xl flex flex-col gap-8">
        {CurrentPackage.modules?.map((mod) => (
          <ModuleItem key={mod.name} module={mod} />
        ))}
      </div>
    </main>
  );
}
