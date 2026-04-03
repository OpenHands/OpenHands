/**
 * Given a model string, extract the provider and model name.
 *
 * Provider assignment for bare model names is handled by the backend before
 * the model list reaches the frontend. This function only parses slash-based
 * ``provider/model`` strings and leaves everything else as a bare model.
 *
 * @example
 * extractModelAndProvider("azure/ada")
 * // returns { provider: "azure", model: "ada", separator: "/" }
 */
export const extractModelAndProvider = (model: string) => {
  const [provider, ...modelId] = model.split("/");

  if (!provider || modelId.length === 0) {
    return { provider: "", model, separator: "" };
  }

  return { provider, model: modelId.join("/"), separator: "/" };
};
