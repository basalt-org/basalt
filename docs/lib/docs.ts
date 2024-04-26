import JSONDocs from "@/docs.json";

type Documentation = {
  decl: Package;
  version: string;
};

export type Package = {
  description: string;
  kind: string;
  modules?: Module[];
  name: string;
  packages?: Package[];
  summary: string;
};

export type Module = {
  aliases?: Alias[];
  description: string;
  functions?: Function[];
  kind: string;
  name: string;
  structs?: Struct[];
  summary: string;
  traits?: string[];
};

export type Alias = {
  description: string;
  kind: string;
  name: string;
  summary: string;
  value: string;
};

export type Function = {
  kind: string;
  name: string;
  overloads?: Overload[];
};

export type Overload = {
  args?: Argument[];
  async: boolean;
  constraints: string;
  description: string;
  isDef: boolean;
  isStatic: boolean;
  kind: string;
  name: string;
  parameters?: Parameter[];
  raises: boolean;
  returnType?: string | null;
  returns: string;
  signature: string;
  summary: string;
};

export type Argument = {
  description: string;
  inout: boolean;
  kind: string;
  name: string;
  owned: boolean;
  passingKind: string;
  type: string;
};

export type Parameter = {
  description: string;
  kind: string;
  name: string;
  type: string;
};

export type Struct = {
  aliases?: Alias[];
  constraints: string;
  description: string;
  fields?: Field[];
  functions?: Function[];
  kind: string;
  name: string;
  parameters?: Parameter[];
  parentTraits?: string[] | null;
  summary: string;
};

export type Field = {
  description: string;
  kind: string;
  name: string;
  summary: string;
  type: string;
};

const Docs: Documentation = JSONDocs;

export type IndexedPackage = {
  package: Package;
  path: string;
};

export type IndexedModule = {
  module: Module;
  path: string;
};

export function getAllPackages(): IndexedPackage[] {
  const packages: IndexedPackage[] = [];

  function walk(pkg: Package, path: string) {
    packages.push({ package: pkg, path });

    if (pkg.packages) {
      pkg.packages.forEach((p) => {
        walk(p, `${path}/${p.name}`);
      });
    }
  }

  walk(Docs.decl, "");
  packages.shift();

  return packages;
}

export function getAllModules(): IndexedModule[] {
  const modules: IndexedModule[] = [];

  function walk(pkg: Package, path: string) {
    if (pkg.modules) {
      pkg.modules.forEach((m) => {
        modules.push({ module: m, path: `${path}#${m.name}` });
      });
    }

    if (pkg.packages) {
      pkg.packages.forEach((p) => {
        walk(p, `${path}/${p.name}`);
      });
    }
  }

  walk(Docs.decl, "");

  return modules;
}

export function findPackage(
  pkg: string[],
  currentPackage: Package = Docs.decl,
): Package | undefined {
  if (pkg.length === 0) {
    return currentPackage;
  }

  const nextPackage = currentPackage.packages?.find((p) => p.name === pkg[0]);
  if (!nextPackage) {
    return undefined;
  }

  return findPackage(pkg.slice(1), nextPackage);
}

export default Docs;
