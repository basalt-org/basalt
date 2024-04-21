import { Documentation } from "@/types/mojo";
import { exec } from "child_process";

export default async function getDocs(): Promise<Documentation> {
    return new Promise((resolve, reject) => {
        exec("mojo doc ../basalt", (error, stdout, stderr) => {
            if (error) {
                reject(error);
            }
            resolve(JSON.parse(stdout));
        });
    });
}
