import { Documentation } from "@/types/mojo";
import { exec } from "child_process";
import { del, list, put } from '@vercel/blob';

export async function getDocs(): Promise<Documentation> {
    return new Promise((resolve, reject) => {
        exec("mojo doc ../basalt", (error, stdout, stderr) => {
            if (error) {
                reject(error);
            }
            resolve(JSON.parse(stdout));
        });
    });
}

export async function loadDocs(): Promise<Documentation> {
    const url = await getDocsUrl();
    const response = await fetch(url);
    return await response.json();
}

export async function saveDocs(docs?: Documentation) {
    await deleteDocs();
    const blob = new Blob([JSON.stringify(await getDocs())], { type: 'application/json' });
    await put('docs', blob, { access: 'public' })
}

export async function getDocsUrl() {
    const blobs = await list();

    if (blobs.blobs.length === 0) {
        await saveDocs();
        return getDocsUrl();
    }

    return blobs.blobs[0].downloadUrl;
}

export async function deleteDocs() {
    del((await list()).blobs.map(blob => blob.url));
}