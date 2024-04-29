import { Alias, Module, Struct, Function } from "@/lib/docs";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Separator } from "./ui/separator";

const AliasValue = ({ alias }: { alias: Alias }) => (
    <span className="text-sm text-primary/90">{alias.name} = {alias.value}</span>
);

const StructValue = ({ struct }: { struct: Struct }) => (
    <span className="text-sm text-primary/90">{struct.name}</span>
);

const FunctionValue = ({ func }: { func: Function }) => (
    <span className="text-sm text-primary/90">{func.name}</span>
);

export default function ModuleCard({ mdl }: { mdl: Module }) {
    return (
        <Card id={mdl.name}>
            <CardHeader>
                <CardTitle className="capitalize">{mdl.name}</CardTitle>
            </CardHeader>
            <CardContent>
                {mdl.aliases && mdl.aliases.length > 0 && (
                    <>
                        <div className="flex flex-col gap-1">
                            <h2 className="text-md text-primary/90">Aliases</h2>
                            <Separator className="w-1/4" />
                            {mdl.aliases.map((alias) => (
                                <AliasValue alias={alias} key={alias.name} />
                            ))}
                        </div>
                        <div className="p-4" />
                    </>
                )}
                {mdl.structs && mdl.structs.length > 0 && (
                    <>
                        <div className="flex flex-col gap-1">
                            <h2 className="text-md text-primary/90">Structs</h2>
                            <Separator className="w-1/4" />
                            {mdl.structs.map((struct) => (
                                <StructValue struct={struct} key={struct.name} />
                            ))}
                        </div>
                        <div className="p-4" />
                    </>
                )}
                {mdl.functions && mdl.functions.length > 0 && (
                    <>
                        <div className="flex flex-col gap-1">
                            <h2 className="text-md text-primary/90">Functions</h2>
                            <Separator className="w-1/4" />
                            {mdl.functions.map((func) => (
                                <FunctionValue func={func} key={func.name} />
                            ))}
                        </div>
                        <div className="p-4" />
                    </>
                )}
            </CardContent>
        </Card>
    )
}