import { extractModelAndProvider } from "./extract-model-and-provider";

/**
 * Given a list of models, organize them by provider.
 * @param models The list of models
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
export const organizeModelsAndProviders = (models: string[]) => {
  const object: Record<string, { separator: string; models: string[] }> = {};

  models.forEach((model) => {
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
  });

  return object;
};
