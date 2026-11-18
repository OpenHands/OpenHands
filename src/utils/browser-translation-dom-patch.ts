/**
 * Browser Translation DOM Patch
 *
 * When browser extensions or built-in browser translation tools (such as Chrome's
 * Google Translate, Edge Translator, or Safari Translate) translate web pages,
 * they mutate live DOM text nodes by wrapping or replacing them with <font> tags.
 *
 * When React Fiber reconciles state changes or unmounts nodes (commitMutationEffects,
 * commitDeletionEffects), React invokes `parent.removeChild(child)` or
 * `parent.insertBefore(newNode, referenceNode)` with its cached DOM references.
 * If the translator has modified the DOM tree, `child.parentNode` is no longer `parent`,
 * causing the browser to throw a fatal DOMException:
 *   "NotFoundError: Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node."
 *
 * This patch intercepts `removeChild` and `insertBefore` on `Node.prototype` to handle
 * these mismatched parent scenarios gracefully, preventing unhandled DOM crashes and
 * subsequent ErrorBoundary crash loops (#19755).
 */

let isPatched = false;
let originalRemoveChild: typeof Node.prototype.removeChild | null = null;
let originalInsertBefore: typeof Node.prototype.insertBefore | null = null;

export function installBrowserTranslationDomPatch(): () => void {
  if (
    typeof window === "undefined" ||
    typeof Node === "undefined" ||
    !Node.prototype
  ) {
    return () => {};
  }

  if (isPatched) {
    return uninstallBrowserTranslationDomPatch;
  }

  originalRemoveChild = Node.prototype.removeChild;
  originalInsertBefore = Node.prototype.insertBefore;

  Node.prototype.removeChild = function <T extends Node>(child: T): T {
    if (child.parentNode !== this) {
      if (child.parentNode) {
        try {
          return child.parentNode.removeChild(child) as T;
        } catch {
          // Swallow if already detached or further mutated
        }
      }
      return child;
    }
    return originalRemoveChild!.call(this, child) as T;
  };

  Node.prototype.insertBefore = function <T extends Node>(
    newNode: T,
    referenceNode: Node | null,
  ): T {
    if (referenceNode && referenceNode.parentNode !== this) {
      if (referenceNode.parentNode) {
        try {
          return referenceNode.parentNode.insertBefore(
            newNode,
            referenceNode,
          ) as T;
        } catch {
          return this.appendChild(newNode) as T;
        }
      }
      return this.appendChild(newNode) as T;
    }
    return originalInsertBefore!.call(this, newNode, referenceNode) as T;
  };

  isPatched = true;
  return uninstallBrowserTranslationDomPatch;
}

export function uninstallBrowserTranslationDomPatch(): void {
  if (!isPatched) return;
  if (originalRemoveChild) {
    Node.prototype.removeChild = originalRemoveChild;
    originalRemoveChild = null;
  }
  if (originalInsertBefore) {
    Node.prototype.insertBefore = originalInsertBefore;
    originalInsertBefore = null;
  }
  isPatched = false;
}
