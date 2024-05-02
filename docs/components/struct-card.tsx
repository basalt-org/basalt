import { Struct } from "@/lib/docs";
import FunctionCard from "./function-card";


export default function StructCard({ struct }: { struct: Struct }) {
    return (
        <>
            <span className="text-md text-primary">{struct.name}</span>
            <ul className="pl-8">
                <li>
                    <span className="text-primary/90">{struct.description}</span>
                </li>
                {struct.parameters && struct.parameters.length > 0 && (
                    <li>
                        <span className="text-primary/90">Parameters</span>
                        <ul className="pl-4">
                            {struct.parameters.map((param) => (
                                <li key={param.name}>
                                    <span className="text-primary/90">{param.name} - {param.type}</span>
                                </li>
                            ))}
                        </ul>
                    </li>
                )}
                <div className="p-2" />
                {struct.fields && struct.fields.length > 0 && (
                    <li>
                        <span className="text-primary/90">Fields</span>
                        <ul className="pl-4">
                            {struct.fields.map((field) => (
                                <li key={field.name}>
                                    <span className="text-primary/90">{field.name} - {field.type}</span>
                                </li>
                            ))}
                        </ul>
                    </li>
                )}
                <div className="p-2" />
                {struct.functions && struct.functions.length > 0 && (
                    <li>
                        <span className="text-primary/90">Functions</span>
                        <ul className="pl-4">
                            {struct.functions.map((func) => (
                                <li key={func.name}>
                                    <FunctionCard func={func} />
                                </li>
                            ))}
                        </ul>
                    </li>
                )}
                <div className="p-2" />
                {struct.aliases && struct.aliases.length > 0 && (
                    <li>
                        <span className="text-primary/90">Aliases</span>
                        <ul className="pl-4">
                            {struct.aliases.map((alias) => (
                                <li key={alias.name}>
                                    <span className="text-primary/90">{alias.name} = {alias.value}</span>
                                </li>
                            ))}
                            </ul>
                    </li>
                )}
            </ul>
        </>
    )
}