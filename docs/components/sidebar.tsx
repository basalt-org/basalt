import { getAllModules, getAllPackages, Package } from "@/lib/docs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "./ui/accordion";
import Link from "next/link";

const modules = getAllModules();
const packages = getAllPackages();

export default function Sidebar({pkg}: {pkg: Package}) {
    return (
        <aside>
            <Accordion type="single" collapsible>
                <AccordionItem value="Packages" className="w-fit px-4">
                    <AccordionTrigger>Packages</AccordionTrigger>
                    <AccordionContent>
                        {pkg.packages?.map((p) => (
                            <>...</>
                        ))}
                    </AccordionContent>
                </AccordionItem>
            </Accordion>
        </aside>
    )
}