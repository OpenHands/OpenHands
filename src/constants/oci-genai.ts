import type { SettingsFieldSchema } from "#/types/settings";

export const OCI_GENAI_PROVIDER = "oci_genai";
export const OCI_REGION_KEY = "llm.oci_region";
export const OCI_PROJECT_ID_KEY = "llm.oci_project_id";
export const OCI_MODELS_DOCS_URL =
  "https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm";
export const OCI_PROJECT_DOCS_URL =
  "https://docs.oracle.com/en-us/iaas/Content/generative-ai/use-project.htm";
export const OCI_API_KEYS_DOCS_URL =
  "https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm";

const OCI_REGION_FIELD: SettingsFieldSchema = {
  key: OCI_REGION_KEY,
  label: "OCI region",
  description: "OCI region used by the Generative AI endpoint.",
  section: "llm",
  section_label: "LLM",
  value_type: "string",
  default: null,
  choices: [],
  depends_on: [],
  prominence: "critical",
  secret: false,
  required: false,
};

const OCI_PROJECT_ID_FIELD: SettingsFieldSchema = {
  key: OCI_PROJECT_ID_KEY,
  label: "OCI Generative AI project OCID",
  description: "Project OCID sent with OCI Generative AI requests.",
  section: "llm",
  section_label: "LLM",
  value_type: "string",
  default: null,
  choices: [],
  depends_on: [],
  prominence: "critical",
  secret: false,
  required: false,
};

export const OCI_GENAI_SCHEMA_FIELDS = [OCI_REGION_FIELD, OCI_PROJECT_ID_FIELD];

export const isOciGenaiModel = (model: unknown): model is string =>
  typeof model === "string" && model.startsWith(`${OCI_GENAI_PROVIDER}/`);
