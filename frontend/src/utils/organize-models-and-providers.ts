import { extractModelAndProvider } from "./extract-model-and-provider";

/**
 * Given a list of models, organize them by provider.
 * @param models The list of model id strings (non-arrays and non-string entries are ignored)
 * @returns An object containing the provider and models
 *
 * @example
 * const models = ["azure/ada", "azure/gpt-35-turbo", "gpt-4o"];
 *
 * organizeModelsAndProviders(models);
 * // returns {
 * //   azure: {
 * //     separator: "/",
 * //     models: ["ada", "gpt-35-turbo"],
 * //   },
 * //   other: {
 * //     separator: "",
 * //     models: ["gpt-4o"],
 * //   },
 * // }
 */
export const organizeModelsAndProviders = (
  models: unknown,
): Record<string, { separator: string; models: string[] }> => {
  const object: Record<string, { separator: string; models: string[] }> = {};

  if (!Array.isArray(models)) {
    return object;
  }

  for (const model of models) {
    if (typeof model === "string") {
      const {
        separator,
        provider,
        model: modelId,
      } = extractModelAndProvider(model);

      const key = provider || "other";
      if (!object[key]) {
        object[key] = { separator, models: [] };
      }
      object[key].models.push(modelId);
    }
  }

  return object;
};
