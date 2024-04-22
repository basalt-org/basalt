import { loadDocs } from "@/lib/docs";

export default async function Module({
  params,
}: {
  params: { module: string };
}) {
  const docs = await loadDocs();
  const module = docs.decl.packages
    .flatMap((p) => p.modules)
    .find((m) => m.name === params.module.toLowerCase())!;

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
              <li key={f.name}>{f.name}</li>
            ))}
          </ul>
        </div>
      ))}

      <h2>Functions</h2>
      {module.functions.map((f) => (
        <div key={f.name}>
          <h3>{f.name}</h3>
          <ul>
            {f.overloads.map((o) => (
              <li key={o.name}>
                <h4>{o.name}</h4>
                <p>{o.summary}</p>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
