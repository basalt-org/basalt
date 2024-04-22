import { BASE_URL } from "@/utils/constants";
import { Documentation } from "@/types/mojo";

export async function getDocs(): Promise<Documentation> {
  const response = await fetch(`${BASE_URL}/api/docs?action=get`, {
    cache: "no-store",
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error);
  }
  return data;
}

export async function loadDocs(): Promise<Documentation> {
  const response = await fetch(`${BASE_URL}/api/docs?action=url`);
  const { url } = await response.json();
  const docsResponse = await fetch(url);
  return await docsResponse.json();
}

export async function saveDocs() {
  await fetch(`${BASE_URL}/api/docs`, { method: "POST" });
}

export async function deleteDocs() {
  await fetch(`${BASE_URL}/api/docs`, { method: "DELETE" });
}
