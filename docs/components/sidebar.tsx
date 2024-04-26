import { Package } from "@/lib/docs";
import { Accordion, AccordionItem, AccordionTrigger } from "./ui/accordion";

export default function Sidebar({pkg}: {pkg: Package}) {
    return (
        <aside>
            <Accordion type="single" collapsible>
                <AccordionItem value="Packages" className="w-fit px-4">
                    <AccordionTrigger>Packages</AccordionTrigger>
                </AccordionItem>
            </Accordion>
        </aside>
    )
}