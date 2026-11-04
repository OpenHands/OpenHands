import { act, renderHook } from "@testing-library/react";
import { createInstance } from "i18next";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { getI18n, setI18n } from "#/i18n";
import { useSyncDocumentLanguage } from "#/hooks/use-sync-document-language";

// Screen readers pick their voice from <html lang>; right-to-left scripts
// also need <html dir> (#17909).
describe("useSyncDocumentLanguage", () => {
  beforeEach(async () => {
    const instance = createInstance();
    await instance.init({ lng: "de", resources: {} });
    setI18n(instance);
    document.documentElement.lang = "en";
    document.documentElement.removeAttribute("dir");
  });

  afterEach(() => {
    setI18n(null);
    document.documentElement.lang = "en";
    document.documentElement.removeAttribute("dir");
  });

  it("applies the active language on mount", () => {
    renderHook(() => useSyncDocumentLanguage());

    expect(document.documentElement.lang).toBe("de");
    expect(document.documentElement.dir).toBe("ltr");
  });

  it("follows language changes, setting rtl for Arabic", async () => {
    renderHook(() => useSyncDocumentLanguage());

    await act(() => getI18n().changeLanguage("ar"));
    expect(document.documentElement.lang).toBe("ar");
    expect(document.documentElement.dir).toBe("rtl");

    await act(() => getI18n().changeLanguage("fr"));
    expect(document.documentElement.lang).toBe("fr");
    expect(document.documentElement.dir).toBe("ltr");
  });

  it("stops following language changes after unmount", async () => {
    const { unmount } = renderHook(() => useSyncDocumentLanguage());
    unmount();

    await act(() => getI18n().changeLanguage("ar"));

    expect(document.documentElement.lang).toBe("de");
    expect(document.documentElement.dir).toBe("ltr");
  });
});
