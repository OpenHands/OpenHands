import { extractModelAndProvider } from "./extract-model-and-provider";

const isEquivalentOpenAIModel = (left: string, right: string) => {
  const leftParts = extractModelAndProvider(left);
  const rightParts = extractModelAndProvider(right);

  return (
    leftParts.model === rightParts.model &&
    ((leftParts.provider === "openai" && !rightParts.provider) ||
      (!leftParts.provider && rightParts.provider === "openai"))
  );
};

/**
 * Check if a model is a custom model. A custom model is a model that is not part of the default models.
 * @param models Full list of models
 * @param model Model to check
 * @returns Whether the model is a custom model
 */
export const isCustomModel = (models: string[], model: string): boolean => {
  if (!model) return false;

  const isKnownModel = models.some(
    (availableModel) =>
      availableModel === model ||
      isEquivalentOpenAIModel(availableModel, model),
  );

  return !isKnownModel;
};
