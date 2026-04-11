/**
 * whisk_generate.mjs
 * ==================
 * Full Whisk pipeline using the correct module API methods.
 *
 * Correct API (from repo examples):
 *   - project.projectId              (not project.id)
 *   - project.addSubject(base64)     (not uploadReference)
 *   - project.generateImageWithReferences(prompt)  (not generateImage with subject)
 *   - imageToBase64(path)            (utility from the package)
 *   - generated.refine(prompt)       (same as before, this one was correct)
 *
 * Usage (called from Python via subprocess):
 *   node whisk_generate.mjs full <imagePath> <prompt> <outputDir>
 *
 * Cookie is read from WHISK_COOKIE environment variable.
 *
 * Output lines parsed by Python:
 *   [+] PROJECT_ID: <uuid>
 *   [+] SUBJECT: added
 *   [+] GENERATED_ID: <id>
 *   [+] GENERATED_PATH: <path>
 *   [+] REFINED_PATH: <path>
 *   [!] ERROR: <message>   → exits with code 1
 */

import { Whisk } from "@rohitaryal/whisk-api";
import path from "path";
import fs from "fs";

// Manual base64 conversion — imageToBase64 is not exported by this version
function imageToBase64(imagePath) {
    const buffer = fs.readFileSync(imagePath);
    return buffer.toString("base64");
}

// ─── Cookie ────────────────────────────────────────────────────

const cookie = process.env.WHISK_COOKIE;
if (!cookie) {
    console.error("[!] ERROR: WHISK_COOKIE environment variable not set");
    process.exit(1);
}

// ─── Args ──────────────────────────────────────────────────────

const [,, mode, ...args] = process.argv;

if (!mode) {
    console.error("[!] ERROR: No mode specified. Use: full");
    process.exit(1);
}

// ─── Helpers ───────────────────────────────────────────────────

function ensureDir(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

// ─── Mode: full ────────────────────────────────────────────────
// newProject → addSubject(base64) → generateImageWithReferences → refine

async function modeFull(imagePath, prompt, outputDir) {
    ensureDir(outputDir);

    if (!fs.existsSync(imagePath)) {
        console.error(`[!] ERROR: Image not found: ${imagePath}`);
        process.exit(1);
    }

    const whisk = new Whisk(cookie);

    // Step 1: Create project
    const project = await whisk.newProject(`char_${Date.now()}`);
    console.log(`[+] PROJECT_ID: ${project.projectId}`);

    // Step 2: Convert character image to base64 and add as SUBJECT
    // addSubject handles caption generation + upload internally
    const base64 = await imageToBase64(imagePath);
    await project.addSubject(base64);
    console.log(`[+] SUBJECT: added`);

    // Step 3: Generate image using subject reference
    // generateImageWithReferences uses whatever subjects/scenes/styles
    // have been added to the project
    const generated = await project.generateImageWithReferences(prompt);
    console.log(`[+] GENERATED_ID: ${generated.id || "unknown"}`);

    const genPath = generated.save(outputDir);
    console.log(`[+] GENERATED_PATH: ${genPath}`);

    // Step 4: Refine the generated image with scene prompt
    const refined = await generated.refine(prompt);
    const refinedPath = refined.save(outputDir);
    console.log(`[+] REFINED_PATH: ${refinedPath}`);
}

// ─── Dispatch ──────────────────────────────────────────────────

try {
    switch (mode) {
        case "full":
            if (args.length < 2) {
                console.error("[!] ERROR: full mode requires: <imagePath> <prompt> [outputDir]");
                process.exit(1);
            }
            await modeFull(args[0], args[1], args[2] || "./output");
            break;

        default:
            console.error(`[!] ERROR: Unknown mode: ${mode}. Use: full`);
            process.exit(1);
    }
} catch (err) {
    const msg = (err.message || String(err)).toLowerCase();
    if (
        msg.includes("401") ||
        msg.includes("unauthenticated") ||
        msg.includes("new cookie") ||
        msg.includes("cookie") ||
        msg.includes("unauthorized")
    ) {
        console.error(`[!] ERROR: 401 Auth failure — ${err.message || err}`);
    } else {
        console.error(`[!] ERROR: ${err.message || err}`);
    }
    process.exit(1);
}
