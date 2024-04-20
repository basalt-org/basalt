import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"

export default function Navbar() {
    return (
        <nav className="flex items-center justify-between p-4 bg-gray-800">
            <div className="flex items-center">
                <Button>Docs</Button>
                <Separator orientation="vertical" className="mx-4 h-8 bg-gradient-to-t from-gray-700 to-slate-600" />
            </div>
        </nav>
    );
}