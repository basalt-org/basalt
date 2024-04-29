"use client";

import { LaptopIcon, MoonIcon, SunIcon } from "lucide-react";
import { getAllPackages, getAllModules } from "@/lib/docs";
import { useCallback, useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandSeparator,
} from "@/components/ui/command";
import { Button } from "./ui/button";
import { useRouter } from "next/navigation";

export default function SearchBar() {
  const { setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const allPackages = getAllPackages();
  const allModules = getAllModules();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || e.key === "/") {
        if (
          (e.target instanceof HTMLElement && e.target.isContentEditable) ||
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement ||
          e.target instanceof HTMLSelectElement
        ) {
          return;
        }

        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const runCommand = useCallback((command: () => unknown) => {
    setOpen(false);
    command();
  }, []);

  return (
    <>
      <Button
        variant="outline"
        className={cn(
          "relative h-8 w-full justify-start rounded-[0.5rem] bg-background text-sm font-normal text-muted-foreground shadow-none sm:pr-12 md:w-40 lg:w-64",
        )}
        onClick={() => setOpen(true)}
      >
        <span className="hidden lg:inline-flex">Search documentation...</span>
        <span className="inline-flex lg:hidden">Search...</span>
        <kbd className="pointer-events-none absolute right-[0.3rem] top-[0.3rem] hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>
      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Type a command or search..." />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>
          <CommandGroup heading="Packages">
            {allPackages.map((pkg) => (
              <CommandItem
                key={pkg.path}
                onSelect={() =>
                  runCommand(() => router.push("/docs" + pkg.path))
                }
                className="capitalize !pointer-events-auto !opacity-80 !cursor-pointer"
              >
                {pkg.package.name}
              </CommandItem>
            ))}
          </CommandGroup>
          <CommandSeparator />
          <CommandGroup heading="Modules">
            {allModules.map((mod) => (
              <CommandItem
                key={mod.path}
                onSelect={() =>
                  runCommand(() => router.push("/docs" + mod.path))
                }
                className="capitalize !pointer-events-auto !opacity-80 !cursor-pointer"
              >
                {mod.module.name}
              </CommandItem>
            ))}
          </CommandGroup>
          <CommandSeparator />
          <CommandGroup heading="Theme">
            <CommandItem
              onSelect={() => runCommand(() => setTheme("light"))}
              className="capitalize !pointer-events-auto !opacity-80 !cursor-pointer"
            >
              <SunIcon className="mr-2 h-4 w-4" />
              Light
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => setTheme("dark"))}
              className="capitalize !pointer-events-auto !opacity-80 !cursor-pointer"
            >
              <MoonIcon className="mr-2 h-4 w-4" />
              Dark
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => setTheme("system"))}
              className="capitalize !pointer-events-auto !opacity-80 !cursor-pointer"
            >
              <LaptopIcon className="mr-2 h-4 w-4" />
              System
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  );
}
