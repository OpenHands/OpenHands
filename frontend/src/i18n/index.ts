import i18n from "i18next";
import Backend from "i18next-http-backend";
import LanguageDetector from "i18next-browser-languagedetector";
import { initReactI18next } from "react-i18next";

export const AvailableLanguages = [
  { label: "English", value: "en" },
  { label: "日本語", value: "ja" },
  { label: "简体中文", value: "zh-CN" },
  { label: "繁體中文", value: "zh-TW" },
  { label: "한국어", value: "ko-KR" },
  { label: "Norsk", value: "no" },
  { label: "Arabic", value: "ar" },
  { label: "Deutsch", value: "de" },
  { label: "Français", value: "fr" },
  { label: "Italiano", value: "it" },
  { label: "Português", value: "pt" },
  { label: "Español", value: "es" },
  { label: "Català", value: "ca" },
  { label: "Türkçe", value: "tr" },
  { label: "Українська", value: "uk" },
];

i18n
  .use(Backend)
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: "en",
    debug: import.meta.env.NODE_ENV === "development",

    // Define supported languages explicitly to prevent 404 errors
    // According to i18next documentation, this is the recommended way to prevent
    // 404 requests for unsupported language codes like 'en-US@posix'
    supportedLngs: AvailableLanguages.map((lang) => lang.value),

    // Do NOT set nonExplicitSupportedLngs: true as it causes 404 errors
    // for region-specific codes not in supportedLngs (per i18next developer)
    nonExplicitSupportedLngs: false,
  });

export default i18n;
