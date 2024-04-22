import { loadDocs } from "@/lib/docs";

export default async function Module({ params }: { params: { module: string } }) {
    const docs = await loadDocs();
    // The module is nxted in a package, find the package that has the module in its list of modules, that is the correct route
    const module = docs.decl.packages.find((p) => p.modules.some((m) => m.name === params.module.toLowerCase()))!.modules.find((m) => m.name === params.module.toLowerCase())!;

    /*
    example module
    {
  aliases: [],
  description: '',
  functions: [
    { kind: 'function', name: 'f64_to_bytes', overloads: [Array] },
    { kind: 'function', name: 'bytes_to_f64', overloads: [Array] }
  ],
  kind: 'module',
  name: 'bytes',
  structs: [
    {
      aliases: [],
      constraints: '',
      description: '',
      fields: [Array],
      functions: [Array],
      kind: 'struct',
      name: 'Bytes',
      parameters: [Array],
      parentTraits: [Array],
      summary: 'Static sequence of bytes.'
    }
  ],
  summary: '',
  traits: []
}
    */

    // Render info with shadcn cards / code

    return (
        <div>
            <h1>{module.name}</h1>
            <p>{module.description}</p>
            <h2>Structs</h2>
            {module.structs.map((s) => (
                <div key={s.name}>
                    <h3>{s.name}</h3>
                    <p>{s.summary}</p>
                    <h4>Fields</h4>
                    <ul>
                        {s.fields.map((f) => (
                            <li key={f.name}>
                                <h5>{f.name}</h5>
                                <p>{f.summary}</p>
                            </li>
                        ))}
                    </ul>
                    <h4>Functions</h4>
                    <ul>
                        {s.functions.map((f) => (
                            <li key={f.name}>
                                <h5>{f.name}</h5>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    );
}
  