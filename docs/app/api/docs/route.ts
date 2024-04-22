import { exec } from "child_process";
import { NextResponse } from "next/server";
import { del, list, put } from "@vercel/blob";

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const action = searchParams.get("action");

  if (action === "get") {
    return new Promise((resolve, reject) => {
      exec("mojo doc ../basalt", (error, stdout, stderr) => {
        if (error) {
          resolve(NextResponse.json({ error: error.message }, { status: 500 }));
        } else {
          resolve(NextResponse.json(JSON.parse(stdout)));
        }
      });
    });
  } else if (action === "url") {
    const blobs = await list();
    if (blobs.blobs.length === 0) {
      await saveDocs();
      return NextResponse.redirect(new URL(request.url));
    }
    return NextResponse.json({ url: blobs.blobs[0].downloadUrl });
  } else {
    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  }
}

export async function POST(): Promise<NextResponse> {
  await saveDocs();
  return NextResponse.json({ message: "Docs saved successfully" });
}

export async function DELETE(): Promise<NextResponse> {
  await deleteDocs();
  return NextResponse.json({ message: "Docs deleted successfully" });
}

async function saveDocs() {
  await deleteDocs();
  const response = await fetch("http://localhost:3000/api/docs?action=get", {
    cache: "no-store",
  });
  const docs = await response.json();
  const blob = new Blob([JSON.stringify(docs)], { type: "application/json" });
  await put("docs", blob, { access: "public" });
}

async function deleteDocs() {
  await del((await list()).blobs.map((blob) => blob.url));
}
