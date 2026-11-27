const fs = require("fs");
const path = require("path");
const i18n = require("../src/i18n/translation.json");

const namespace = "openhands";

// { [lang]: { [key]: content } }
const translationMap = {};

Object.entries(i18n).forEach(([key, transMap]) => {
  Object.entries(transMap).forEach(([lang, content]) => {
    if (!translationMap[lang]) {
      translationMap[lang] = {};
    }
    translationMap[lang][key] = content;
  });
});

// remove old locales directory
const localesPath = path.join(__dirname, "../public/locales");
if (fs.existsSync(localesPath)) {
  fs.rmSync(localesPath, { recursive: true });
}

// write translation files
Object.entries(translationMap).forEach(([lang, transMap]) => {
  const filePath = path.join(
    __dirname,
    `../public/locales/${lang}/${namespace}.json`,
  );
  if (!fs.existsSync(filePath)) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
  }
  fs.writeFileSync(filePath, JSON.stringify(transMap, null, 2));
});

// write translation key enum
const transKeys = Object.keys(translationMap.en);
const transKeyDeclareFilePath = path.join(
  __dirname,
  "../src/i18n/declaration.ts",
);
if (!fs.existsSync(transKeyDeclareFilePath)) {
  fs.mkdirSync(path.dirname(transKeyDeclareFilePath), { recursive: true });
}
fs.writeFileSync(
  transKeyDeclareFilePath,
  `
// this file generate by script, don't modify it manually!!!
export enum I18nKey {
${transKeys.map((key) => `  ${key} = "${key}",`).join("\n")}
}`.trim() + "\n",
);
