import Link from "next/link";
import Image from "next/image";
import { Github } from "lucide-react";
import { ThemeToggle } from "./theme-toggle";
import SearchBar from "./searchbar";

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2">
      <Image src="/basalt.png" alt="Basalt" width={40} height={40} />
      <span className="text-lg font-medium text-primary/90">Basalt</span>
    </Link>
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
      <span className="hidden md:inline">GitHub</span>
    </Link>
  );
}

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full shadow-sm border-b bg-white dark:bg-black">
      <div className="container flex h-16 items-center justify-between px-4 md:px-6">
        <Logo />
        <div className="p-4 flex flex-1 justify-center max-w-lg mx-auto">
          <SearchBar />
        </div>
        <div className="inline-flex">
          <GitHubLink />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
