import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  installBrowserTranslationDomPatch,
  uninstallBrowserTranslationDomPatch,
} from "#/utils/browser-translation-dom-patch";

describe("browser-translation-dom-patch (#19755)", () => {
  let container: HTMLDivElement;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    installBrowserTranslationDomPatch();
  });

  afterEach(() => {
    uninstallBrowserTranslationDomPatch();
    container.remove();
  });

  describe("removeChild resilience", () => {
    it("removes child normally when parentNode matches", () => {
      const child = document.createElement("span");
      container.appendChild(child);

      expect(child.parentNode).toBe(container);
      const removed = container.removeChild(child);

      expect(removed).toBe(child);
      expect(child.parentNode).toBeNull();
      expect(container.contains(child)).toBe(false);
    });

    it("safely handles child wrapped by browser translator font tags without throwing NotFoundError", () => {
      // Simulate Google Translate injecting <font> around a node
      const fontWrapper = document.createElement("font");
      const textNode = document.createTextNode("Translated text");
      fontWrapper.appendChild(textNode);
      container.appendChild(fontWrapper);

      // In real life, React Fiber retains cached reference to `textNode` and calls `container.removeChild(textNode)`.
      // Before this patch, `container.removeChild(textNode)` would throw:
      // "NotFoundError: The node to be removed is not a child of this node."
      expect(textNode.parentNode).toBe(fontWrapper);
      expect(textNode.parentNode).not.toBe(container);

      expect(() => {
        const removed = container.removeChild(textNode);
        expect(removed).toBe(textNode);
      }).not.toThrow();

      expect(textNode.parentNode).toBeNull();
      expect(fontWrapper.contains(textNode)).toBe(false);
    });

    it("safely handles completely detached child without throwing", () => {
      const detachedNode = document.createElement("div");
      expect(detachedNode.parentNode).toBeNull();

      expect(() => {
        const result = container.removeChild(detachedNode);
        expect(result).toBe(detachedNode);
      }).not.toThrow();
    });
  });

  describe("insertBefore resilience", () => {
    it("inserts normally when referenceNode matches parentNode", () => {
      const refNode = document.createElement("span");
      container.appendChild(refNode);

      const newNode = document.createElement("b");
      const result = container.insertBefore(newNode, refNode);

      expect(result).toBe(newNode);
      expect(newNode.nextSibling).toBe(refNode);
      expect(container.firstChild).toBe(newNode);
    });

    it("safely handles reparented referenceNode without throwing NotFoundError", () => {
      // Simulate reference node reparented by translator inside a <font> tag
      const fontWrapper = document.createElement("font");
      const refNode = document.createTextNode("Ref text");
      fontWrapper.appendChild(refNode);
      container.appendChild(fontWrapper);

      const newNode = document.createTextNode("New text");

      expect(() => {
        const result = container.insertBefore(newNode, refNode);
        expect(result).toBe(newNode);
      }).not.toThrow();

      // Node should be safely inserted before refNode in its actual parent
      expect(newNode.parentNode).toBe(fontWrapper);
      expect(newNode.nextSibling).toBe(refNode);
    });

    it("falls back to appendChild if referenceNode is detached", () => {
      const detachedRefNode = document.createElement("span");
      const newNode = document.createElement("div");

      expect(() => {
        const result = container.insertBefore(newNode, detachedRefNode);
        expect(result).toBe(newNode);
      }).not.toThrow();

      expect(container.contains(newNode)).toBe(true);
    });
  });

  describe("lifecycle and idempotency", () => {
    it("is idempotent when installed multiple times", () => {
      const uninstall1 = installBrowserTranslationDomPatch();
      const uninstall2 = installBrowserTranslationDomPatch();

      expect(typeof uninstall1).toBe("function");
      expect(typeof uninstall2).toBe("function");

      uninstall1();
      // Original prototype methods should be restored
      const detached = document.createElement("span");
      expect(() => container.removeChild(detached)).toThrow();
    });

    it("cleanly uninstalls and restores native throw behavior", () => {
      uninstallBrowserTranslationDomPatch();

      const foreignChild = document.createElement("span");
      const otherContainer = document.createElement("div");
      otherContainer.appendChild(foreignChild);

      expect(() => container.removeChild(foreignChild)).toThrowError(
        /not a child of this node/i,
      );
    });
  });
});
