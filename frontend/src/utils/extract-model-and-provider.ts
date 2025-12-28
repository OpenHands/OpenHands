import { isNumber } from "./is-number";
import {
  VERIFIED_ANTHROPIC_MODELS,
  VERIFIED_MISTRAL_MODELS,
  VERIFIED_OPENAI_MODELS,
  VERIFIED_OPENHANDS_MODELS,
} from "./verified-models";

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
 * Given a model string, extract the provider and model name. Currently the supported separators are "/" and "."
 * @param model The model string
 * @returns An object containing the provider, model name, and separator
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
    // no "/" or "." separator found
    if (VERIFIED_OPENAI_MODELS.includes(split[0])) {
      return { provider: "openai", model: split[0], separator: "/" };
    }
    if (VERIFIED_ANTHROPIC_MODELS.includes(split[0])) {
      return { provider: "anthropic", model: split[0], separator: "/" };
    }
    if (VERIFIED_MISTRAL_MODELS.includes(split[0])) {
      return { provider: "mistral", model: split[0], separator: "/" };
    }
    if (VERIFIED_OPENHANDS_MODELS.includes(split[0])) {
      return { provider: "openhands", model: split[0], separator: "/" };
    }
    // return as model only
    return { provider: "", model, separator: "" };
  }
  const [provider, ...modelId] = split;
  return { provider, model: modelId.join(separator), separator };
};
