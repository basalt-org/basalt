import Docs from "@/docs.json"

export default function Package({ params }: { params: { pkg: string } }) {
    const pkg = Docs.decl.packages.find((pkg) => pkg.name === params.pkg)

    return (
        <main className="flex items-center justify-center pt-32">
            
        </main>
    )
}