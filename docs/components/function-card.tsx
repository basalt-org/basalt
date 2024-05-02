"use client";

import { Function } from "@/lib/docs";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Dialog, DialogContent, DialogTrigger } from "./ui/dialog";

export default async function FunctionCard({ func }: { func: Function }) {
  const { name, overloads } = func;
  const { constraints, isStatic, summary } = overloads![0];
  const signatures = overloads!.map((overload) => overload.signature);

  return (
    <Dialog>
      <div>
        <DialogTrigger>{name}</DialogTrigger>
      </div>
      <DialogContent className="sm:max-w-md">
        <Card>
          <CardHeader>
            <CardTitle>{name}</CardTitle>
            {summary && <p>{summary}</p>}
            {isStatic && <p>Static</p>}
            {constraints && <p>{constraints}</p>}
          </CardHeader>
          <CardContent>
            <ul>
              {signatures.map((signature) => (
                <li key={signature} className={
                  "p-2 my-2 bg-gray-100 dark:bg-gray-800 rounded-md"
                }>
                  <code>{signature}</code>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </DialogContent>
    </Dialog>
  );
}
