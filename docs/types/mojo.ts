export type Documentation = {
    decl: {
        description: string;
        kind: "package";
        modules: Module[];
        name: string;
        packages: Package[];
    };
};

export type Package = {
    description: string;
    kind: "package";
    modules: Module[];
};

export type Module = {
    aliases: Alias[];
    description: string;
    functions: Fn[];
    kind: "module";
    name: string;
    structs: Struct[];
    summary: string;
    traits: Trait[];
};

export type Alias = {
    description: string;
    kind: "alias";
    name: string;
    summary: string;
    value: string;
};

export type Fn = {
    kind: "function";
    name: string;
    overloads: Overload[];
};

export type Overload = {
    args: Argument[];
    async: boolean;
    constraints: string;
    description: string;
    isDef: boolean;
    isStatic: boolean;
    kind: "function";
    name: string;
    parameters: Parameter[];
    raises: boolean;
    returntype: string | null;
    returns: string;
    signature: string;
    summary: string;
};

export type Argument = {
    description: string;
    inout: boolean;
    kind: "argument";
    name: string;
    owned: boolean;
    passingKind: string;
    type: string;
};

export type Parameter = {
    description: string;
    kind: "parameter";
    name: string;
    type: string;
};

export type Struct = {
    aliases: any[];
    constraints: string;
    description: string;
    fields: Field[];
    functions: Fn[];
    kind: "struct";
    name: string;
    parameters: any[];
    parentTraits: string[];
    summary: string;
};

export type Field = {
    description: string;
    kind: "field";
    name: string;
    summary: string;
    type: string;
};

export type Trait = {
    // Add trait properties if needed
};
