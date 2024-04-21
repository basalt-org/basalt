import getDocs from "@/lib/docs";

export default async function Home() {
  console.log(await getDocs());
  return (
    <main>
    </main>
  );
}
