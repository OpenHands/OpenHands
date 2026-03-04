import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel

from openhands.core.config import LLMConfig
from openhands.llm.llm import LLM


class Feature(BaseModel):
    """Represents a single boolean feature that can be extracted from issue descriptions.

    Features are semantic properties of issues (e.g., "has_code_example", "requires_debugging")
    that are evaluated by LLMs and used as input to the solvability classifier.
    """

    identifier: str
    """Unique identifier for the feature, used as column name in feature matrices."""

    description: str
    """Human-readable description of what the feature represents, used in LLM prompts."""

    @property
    def to_tool_description_field(self) -> dict[str, Any]:
        """Convert this feature to a JSON schema field for LLM tool calling.

        Returns:
            dict: JSON schema field definition for this feature.
        """
        return {
            'type': 'boolean',
            'description': self.description,
        }


class EmbeddingDimension(BaseModel):
    """Represents a single dimension (feature evaluation) within a feature embedding sample.

    Each dimension corresponds to one feature being evaluated as true/false for a given issue.
    """

    feature_id: str
    """Identifier of the feature being evaluated."""

    result: bool
    """Boolean result of the feature evaluation for this sample."""


# Type alias for a single embedding sample - maps feature identifiers to boolean values
EmbeddingSample = dict[str, bool]
"""
A single sample from the LLM evaluation of features for an issue.
Maps feature identifiers to their boolean evaluations.
"""


class FeatureEmbedding(BaseModel):
    """Represents the complete feature embedding for a single issue, including multiple samples
    and associated metadata about the LLM calls used to generate it.

    Multiple samples are collected to account for LLM variability and provide more robust
    feature estimates through averaging.
    """

    samples: list[EmbeddingSample]
    """List of individual feature evaluation samples from the LLM."""

    prompt_tokens: int | None = None
    """Total prompt tokens consumed across all LLM calls for this embedding."""

    completion_tokens: int | None = None
    """Total completion tokens generated across all LLM calls for this embedding."""

    response_latency: float | None = None
    """Total response latency (seconds) across all LLM calls for this embedding."""

    @property
    def dimensions(self) -> list[str]:
        """Get all unique feature identifiers present across all samples.

        Returns:
            list[str]: List of feature identifiers that appear in at least one sample.
        """
        dims: set[str] = set()
        for sample in self.samples:
            dims.update(sample.keys())
        return list(dims)

    def coefficient(self, dimension: str) -> float | None:
        """Calculate the average coefficient (0-1) for a specific feature dimension.

        This computes the proportion of samples where the feature was evaluated as True,
        providing a continuous feature value for the classifier.

        Args:
            dimension: Feature identifier to calculate coefficient for.

        Returns:
            float | None: Average coefficient (0.0-1.0), or None if dimension not found.
        """
        # Extract boolean values for this dimension, converting to 0/1
        values = [
            1 if v else 0
            for v in [sample.get(dimension) for sample in self.samples]
            if v is not None
        ]
        if values:
            return sum(values) / len(values)
        return None

    def to_row(self) -> dict[str, Any]:
        """Convert the embedding to a flat dictionary suitable for DataFrame construction.

        Returns:
            dict[str, Any]: Dictionary with metadata fields and feature coefficients.
        """
        return {
            'response_latency': self.response_latency,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            **{dimension: self.coefficient(dimension) for dimension in self.dimensions},
        }

    def sample_entropy(self) -> dict[str, float]:
        """Calculate the Shannon entropy of feature evaluations across samples.

        Higher entropy indicates more variability in LLM responses for a feature,
        which may suggest ambiguity in the feature definition or issue description.

        Returns:
            dict[str, float]: Mapping of feature identifiers to their entropy values (0-1).
        """
        from collections import Counter
        from math import log2

        entropy = {}
        for dimension in self.dimensions:
            # Count True/False occurrences for this feature across samples
            counts = Counter(sample.get(dimension, False) for sample in self.samples)
            total = sum(counts.values())
            if total == 0:
                entropy[dimension] = 0.0
                continue
            # Calculate Shannon entropy: -Σ(p * log2(p))
            entropy_value = -sum(
                (count / total) * log2(count / total)
                for count in counts.values()
                if count > 0
            )
            entropy[dimension] = entropy_value
        return entropy


class Featurizer(BaseModel):
    """Orchestrates LLM-based feature extraction from issue descriptions.

    The Featurizer uses structured LLM tool calling to evaluate boolean features
    for issue descriptions. It handles prompt construction, tool schema generation,
    and batch processing with concurrency.
    """

    system_prompt: str
    """System prompt that provides context and instructions to the LLM."""

    message_prefix: str
    """Prefix added to user messages before the issue description."""

    features: list[Feature]
    """List of features to extract from each issue description."""

    def system_message(self) -> dict[str, Any]:
        """Construct the system message for LLM conversations.

        Returns:
            dict[str, Any]: System message dictionary for LLM API calls.
        """
        return {
            'role': 'system',
            'content': self.system_prompt,
        }

    def user_message(
        self, issue_description: str, set_cache: bool = True
    ) -> dict[str, Any]:
        """Construct the user message containing the issue description.

        Args:
            issue_description: The description of the issue to analyze.
            set_cache: Whether to enable ephemeral caching for this message.
                      Should be False for single samples to avoid cache overhead.

        Returns:
            dict[str, Any]: User message dictionary for LLM API calls.
        """
        message: dict[str, Any] = {
            'role': 'user',
            'content': f'{self.message_prefix}{issue_description}',
        }
        if set_cache:
            message['cache_control'] = {'type': 'ephemeral'}
        return message

    @property
    def tool_choice(self) -> dict[str, Any]:
        """Get the tool choice configuration for forcing LLM to use the featurizer tool.

        Returns:
            dict[str, Any]: Tool choice configuration for LLM API calls.
        """
        return {
            'type': 'function',
            'function': {'name': 'call_featurizer'},
        }

    @property
    def tool_description(self) -> dict[str, Any]:
        """Generate the tool schema for the featurizer function.

        Creates a JSON schema that describes the featurizer tool with all configured
        features as boolean parameters.

        Returns:
            dict[str, Any]: Complete tool description for LLM API calls.
        """
        return {
            'type': 'function',
            'function': {
                'name': 'call_featurizer',
                'description': 'Record the features present in the issue.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        feature.identifier: feature.to_tool_description_field
                        for feature in self.features
                    },
                },
            },
        }

    def embed(
        self,
        issue_description: str,
        llm_config: LLMConfig,
        temperature: float = 1.0,
        samples: int = 10,
    ) -> FeatureEmbedding:
        """Generate a feature embedding for a single issue description.

        Makes multiple LLM calls to collect samples and reduce variance in feature evaluations.
        Each call uses tool calling to extract structured boolean feature values.

        Args:
            issue_description: The description of the issue to analyze.
            llm_config: Configuration for the LLM to use.
            temperature: Sampling temperature for the model. Higher values increase randomness.
            samples: Number of samples to generate for averaging.

        Returns:
            FeatureEmbedding: Complete embedding with samples and metadata.
        """
        embedding_samples: list[dict[str, Any]] = []
        response_latency: float = 0.0
        prompt_tokens: int = 0
        completion_tokens: int = 0

        # TODO: use llm registry
        llm = LLM(llm_config, service_id='solvability')

        # Generate multiple samples to account for LLM variability
        for _ in range(samples):
            start_time = time.time()
            response = llm.completion(
                messages=[
                    self.system_message(),
                    self.user_message(issue_description, set_cache=(samples > 1)),
                ],
                tools=[self.tool_description],
                tool_choice=self.tool_choice,
                temperature=temperature,
            )
            stop_time = time.time()

            # Extract timing and token usage metrics
            latency = stop_time - start_time
            # Parse the structured tool call response containing feature evaluations
            features = response.choices[0].message.tool_calls[0].function.arguments  # type: ignore[index, union-attr]
            embedding = json.loads(features)

            # Accumulate results and metrics
            embedding_samples.append(embedding)
            prompt_tokens += response.usage.prompt_tokens  # type: ignore[union-attr, attr-defined]
            completion_tokens += response.usage.completion_tokens  # type: ignore[union-attr, attr-defined]
            response_latency += latency

        return FeatureEmbedding(
            samples=embedding_samples,
            response_latency=response_latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def embed_batch(
        self,
        issue_descriptions: list[str],
        llm_config: LLMConfig,
        temperature: float = 1.0,
        samples: int = 10,
    ) -> list[FeatureEmbedding]:
        """Generate embeddings for a batch of issue descriptions using concurrent processing.

        Processes multiple issues in parallel to improve throughput while maintaining
        result ordering.

        Args:
            issue_descriptions: List of issue descriptions to analyze.
            llm_config: Configuration for the LLM to use.
            temperature: Sampling temperature for the model.
            samples: Number of samples to generate per issue.

        Returns:
            list[FeatureEmbedding]: List of embeddings in the same order as input.
        """
        with ThreadPoolExecutor() as executor:
            # Submit all embedding tasks concurrently
            future_to_desc = {
                executor.submit(
                    self.embed,
                    desc,
                    llm_config,
                    temperature=temperature,
                    samples=samples,
                ): i
                for i, desc in enumerate(issue_descriptions)
            }

            # Collect results in original order to maintain consistency
            results: list[FeatureEmbedding] = [None] * len(issue_descriptions)  # type: ignore[list-item]
            for future in as_completed(future_to_desc):
                index = future_to_desc[future]
                results[index] = future.result()

            return results

    def feature_identifiers(self) -> list[str]:
        """Get the identifiers of all configured features.

        Returns:
            list[str]: List of feature identifiers in the order they were defined.
        """
        return [feature.identifier for feature in self.features]
