import type { Module } from "@/lib/docs";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export default function ModuleItem({ module }: { module: Module }) {
  return (
    <Card className="py-4 w-1/2">
      <CardHeader>
        <CardTitle>{module.name}</CardTitle>
      </CardHeader>
      <CardContent>
        <p>{module.summary}</p>
        <p>{module.description}</p>
        {module.traits && module.traits.length > 0 && (
          <>
            <h3 className="font-semibold mt-2">Traits:</h3>
            <ul>
              {module.traits.map((trait) => (
                <li key={trait}>{trait}</li>
              ))}
            </ul>
          </>
        )}
        {module.functions && module.functions.length > 0 && (
          <>
            <h3 className="font-semibold mt-2">Functions:</h3>
            <ul>
              {module.functions.map((func) => (
                <li key={func.name}>{func.name}</li>
              ))}
            </ul>
          </>
        )}
        {module.structs && module.structs.length > 0 && (
          <>
            <h3 className="font-semibold mt-2">Structs:</h3>
            <ul>
              {module.structs.map((struct) => (
                <li key={struct.name}>{struct.name}</li>
              ))}
            </ul>
          </>
        )}
        {module.aliases && module.aliases.length > 0 && (
          <>
            <h3 className="font-semibold mt-2">Aliases:</h3>
            <ul>
              {module.aliases.map((alias) => (
                <li key={alias.name}>
                  {alias.name} - {alias.value}
                </li>
              ))}
            </ul>
          </>
        )}
      </CardContent>
    </Card>
  );
}
