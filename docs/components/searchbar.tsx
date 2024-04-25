"use client";

import { Search } from "lucide-react";
import { Input } from "./ui/input";
import { allModules, allPackages } from "@/lib/docs";

export default function SearchBar() {
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