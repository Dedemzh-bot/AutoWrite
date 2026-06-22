import fs from "node:fs";

const source = fs.readFileSync(new URL("../web_app.py", import.meta.url), "utf8");
const tripleQuote = '"'.repeat(3);
const marker = `HTML_PAGE = r${tripleQuote}`;
const start = source.indexOf(marker);
if (start < 0) {
  throw new Error("找不到 web_app.py 中的 HTML_PAGE");
}
const htmlStart = start + marker.length;
const htmlEnd = source.indexOf(tripleQuote, htmlStart);
const html = source.slice(htmlStart, htmlEnd);
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(
  (match) => match[1],
);
if (!scripts.length) {
  throw new Error("HTML_PAGE 中没有可校验的 script");
}
for (const script of scripts) {
  new Function(script);
}
console.log(`Web JavaScript syntax OK; scripts=${scripts.length}`);
