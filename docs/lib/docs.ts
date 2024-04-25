import JSONDocs from "@/docs.json";

type Documentation = {
  decl: Package;
  version: string;
};

type Package = {
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

function findModules(
  pkg: Package,
  modules: Module[] = [],
): Module[] {
  if (pkg.modules) {
    modules.push(...pkg.modules);
  }

  if (pkg.packages) {
    for (const p of pkg.packages) {
      findModules(p, modules);
    }
  }

  return modules;
}

function findPackages(
  pkg: Package,
  packages: Package[] = [],
): Package[] {
  packages.push(pkg);

  if (pkg.packages) {
    for (const p of pkg.packages) {
      findPackages(p, packages);
    }
  }

  return packages;
}


export const allModules = findModules(Docs.decl);
export const allPackages = findPackages(Docs.decl);

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
