import Link from "next/link";
import Image from "next/image";
import { Input } from "@/components/ui/input";
import { Github, Search } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2">
      <Image src="/basalt.png" alt="Basalt" width={36} height={36} />
      <span className="text-lg font-semibold">Basalt</span>
    </Link>
  );
}

function SearchBar() {
  return (
    <div className="relative w-full">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-500 dark:text-gray-400" />
      <Input
        className="w-full rounded-md border border-gray-200 bg-gray-100 py-2 pl-10 pr-4 text-sm focus:border-gray-400 focus:outline-none focus:ring-0 dark:border-gray-800 dark:bg-gray-800 dark:text-gray-50"
        placeholder="Search documentation..."
        type="search"
      />
    </div>
  );
}

function GitHubLink() {
  return (
    <Link
      href="https://github.com/basalt-org/basalt"
      className="flex items-center gap-2 rounded-md p-2.5 mx-2 text-sm font-medium transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
      target="_blank"
      rel="noopener noreferrer"
    >
      <Github className="h-5 w-5" />
      <span className="hidden sm:inline">GitHub</span>
    </Link>
  );
}

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full bg-white shadow-sm border-b border-gray-200 dark:bg-gray-950 dark:border-gray-700">
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        <Logo />
        <div className="hidden md:flex flex-1 justify-center max-w-lg mx-auto">
          <SearchBar />
        </div>
        <GitHubLink />
        <ThemeToggle />
      </div>
    </header>
  );
}
