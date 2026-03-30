import { isNumber } from "./is-number";

/**
 * Checks if the split array is actually a version number.
 * @param split The split array of the model string
 * @returns Boolean indicating if the split is actually a version number
 *
 * @example
 * const split = ["gpt-3", "5-turbo"] // incorrectly split from "gpt-3.5-turbo"
 * splitIsActuallyVersion(split) // returns true
 */
const splitIsActuallyVersion = (split: string[]) =>
  split[1]?.[0] && isNumber(split[1][0]);

/**
 * Given a model string, extract the provider and model name.
 * Supported separators are "/" and ".".
 *
 * NOTE: Provider assignment for bare model names (e.g. ``gpt-5.2`` →
 * ``openai/gpt-5.2``) is now handled by the backend **before** the model
 * list reaches the frontend.  This function only needs to *parse* the
 * ``provider/model`` string — it no longer carries hardcoded lookup tables.
 *
 * @example
 * extractModelAndProvider("azure/ada")
 * // returns { provider: "azure", model: "ada", separator: "/" }
 *
 * extractModelAndProvider("cohere.command-r-v1:0")
 * // returns { provider: "cohere", model: "command-r-v1:0", separator: "." }
 */
export const extractModelAndProvider = (model: string) => {
  let separator = "/";
  let split = model.split(separator);
  if (split.length === 1) {
    // no "/" separator found, try with "."
    separator = ".";
    split = model.split(separator);
    if (splitIsActuallyVersion(split)) {
      split = [split.join(separator)]; // undo the split
    }
  }
  if (split.length === 1) {
    // No recognised separator — return as bare model.
    return { provider: "", model, separator: "" };
  }
  const [provider, ...modelId] = split;
  return { provider, model: modelId.join(separator), separator };
};
