import Docs from "@/docs.json"

export default function Package({ params }: { params: { pkg: string } }) {
    const CurrentPackage = Docs.decl.packages.find((pkg) => pkg.name === params.pkg);

    const renderList = (title: string, items: any[], itemKey = 'name') => (
        items.length > 0 && (
            <div className="pl-4 py-2">
                <h4 className="text-sm text-primary/90">{title}</h4>
                <ul className="list-disc pl-5">
                    {items.map((item) => (
                        <li key={item[itemKey]}>
                            {item.value ?
                                <span className="text-primary/70">
                                    {item[itemKey]} = {item.value}
                                </span> :
                                <a href={`#${item[itemKey]}`} className="text-primary/70 hover:text-primary/90">
                                    {item[itemKey]}
                                </a>
                            }
                        </li>
                    ))}
                </ul>
            </div>
        )
    );

    return (
        <main className="flex items-center justify-center pt-32">
            <aside className="hidden md:block w-80 h-screen fixed left-0 top-16 border-r border-primary/20 overflow-y-auto dark:text-white text-gray-900">
                {CurrentPackage?.modules.map((mod) => (
                    <div key={mod.name} className="border-y border-gray-300 dark:border-gray-800">
                        <a href={`#${mod.name}`} className="block p-2 text-primary/90 hover:text-primary capitalize">{mod.name}</a>
                        {renderList('Aliases', mod.aliases, 'name')}
                        {renderList('Functions', mod.functions, 'name')}
                        {renderList('Structs', mod.structs, 'name')}
                    </div>
                ))}
                <div className="h-16" />
            </aside>
        </main>
    );
}
