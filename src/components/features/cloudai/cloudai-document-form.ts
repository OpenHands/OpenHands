import type { AppwriteAttribute } from "#/api/integrations/appwrite-service";
import type { CloudAiFormField } from "./cloudai-shared";

const SYSTEM_DOC_KEYS = new Set([
  "$id",
  "$createdAt",
  "$updatedAt",
  "$permissions",
  "$databaseId",
  "$collectionId",
  "$sequence",
]);

export function stripDocumentSystemFields(
  doc: Record<string, unknown>,
): Record<string, unknown> {
  const rest: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(doc)) {
    if (!SYSTEM_DOC_KEYS.has(key)) {
      rest[key] = value;
    }
  }
  return rest;
}

export function valueToFormString(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

export function buildDocumentFormValues(
  attributes: AppwriteAttribute[] | undefined,
  documentData?: Record<string, unknown>,
  includeDocumentId = false,
): Record<string, string> {
  const values: Record<string, string> = {};
  if (includeDocumentId) {
    values.id = "";
  }

  const attrs = (attributes ?? []).filter(
    (attr) => attr.key && attr.status !== "failed",
  );

  if (attrs.length > 0) {
    for (const attr of attrs) {
      const raw = documentData?.[attr.key];
      if (attr.type === "boolean" && !attr.array) {
        values[attr.key] = valueToFormString(
          raw === undefined ? (attr.default ?? false) : raw,
        );
      } else {
        values[attr.key] = valueToFormString(
          raw === undefined ? (attr.default ?? "") : raw,
        );
      }
    }
    return values;
  }

  const data = documentData ? stripDocumentSystemFields(documentData) : {};
  const entries = Object.entries(data);
  if (entries.length === 0) {
    values.__json__ = "{}";
    return values;
  }
  for (const [key, value] of entries) {
    values[key] = valueToFormString(value);
  }
  return values;
}

export function buildDocumentFormFields(
  attributes: AppwriteAttribute[] | undefined,
  documentData: Record<string, unknown> | undefined,
  labels: {
    id: string;
    arrayHelp: string;
    datetimeHelp: string;
    jsonFallback: string;
  },
  includeDocumentId: boolean,
): CloudAiFormField[] {
  const fields: CloudAiFormField[] = [];
  if (includeDocumentId) {
    fields.push({ key: "id", label: labels.id, type: "text" });
  }

  const attrs = (attributes ?? []).filter(
    (attr) => attr.key && attr.status !== "failed",
  );

  if (attrs.length > 0) {
    for (const attr of attrs) {
      fields.push(attributeToFormField(attr, labels));
    }
    return fields;
  }

  const data = documentData ? stripDocumentSystemFields(documentData) : {};
  const keys = Object.keys(data);
  if (keys.length === 0) {
    // Empty create without schema: keep one free-text JSON fallback field.
    fields.push({
      key: "__json__",
      label: labels.jsonFallback,
      multiline: true,
      help: labels.arrayHelp,
    });
    return fields;
  }

  for (const key of keys) {
    const value = data[key];
    if (typeof value === "boolean") {
      fields.push({ key, label: key, checkbox: true });
    } else if (typeof value === "number") {
      fields.push({ key, label: key, type: "number" });
    } else if (typeof value === "object") {
      fields.push({
        key,
        label: key,
        multiline: true,
        help: labels.arrayHelp,
      });
    } else {
      fields.push({ key, label: key, type: "text" });
    }
  }
  return fields;
}

function attributeToFormField(
  attr: AppwriteAttribute,
  labels: { arrayHelp: string; datetimeHelp: string },
): CloudAiFormField {
  const label = attr.key;
  if (attr.array) {
    return {
      key: attr.key,
      label,
      multiline: true,
      required: attr.required,
      help: labels.arrayHelp,
    };
  }

  switch (attr.type) {
    case "boolean":
      return {
        key: attr.key,
        label,
        checkbox: true,
        required: attr.required,
      };
    case "integer":
    case "float":
    case "double":
      return {
        key: attr.key,
        label,
        type: "number",
        required: attr.required,
      };
    case "email":
      return {
        key: attr.key,
        label,
        type: "email",
        required: attr.required,
      };
    case "url":
      return {
        key: attr.key,
        label,
        type: "url",
        required: attr.required,
      };
    case "datetime":
      return {
        key: attr.key,
        label,
        type: "text",
        required: attr.required,
        help: labels.datetimeHelp,
      };
    case "enum":
      return {
        key: attr.key,
        label,
        options: attr.elements ?? [],
        required: attr.required,
      };
    default:
      return {
        key: attr.key,
        label,
        type: "text",
        required: attr.required,
      };
  }
}

export function formValuesToDocumentData(
  values: Record<string, string>,
  attributes: AppwriteAttribute[] | undefined,
): Record<string, unknown> {
  if (values.__json__ !== undefined) {
    return JSON.parse(values.__json__ || "{}") as Record<string, unknown>;
  }

  const attrs = (attributes ?? []).filter(
    (attr) => attr.key && attr.status !== "failed",
  );
  const attrByKey = new Map(attrs.map((attr) => [attr.key, attr]));
  const data: Record<string, unknown> = {};

  for (const [key, raw] of Object.entries(values)) {
    if (key === "id") continue;
    const attr = attrByKey.get(key);
    data[key] = coerceFormValue(raw, attr);
  }
  return data;
}

function coerceFormValue(
  raw: string,
  attr: AppwriteAttribute | undefined,
): unknown {
  if (attr?.array) {
    if (!raw.trim()) return [];
    return JSON.parse(raw) as unknown;
  }

  const type = attr?.type;
  if (type === "boolean") {
    return raw === "true";
  }
  if (type === "integer") {
    if (raw.trim() === "") return null;
    return Number.parseInt(raw, 10);
  }
  if (type === "float" || type === "double") {
    if (raw.trim() === "") return null;
    return Number.parseFloat(raw);
  }

  // Fallback inference when no schema.
  if (!attr) {
    if (raw === "true" || raw === "false") return raw === "true";
    if (
      raw.trim() !== "" &&
      !Number.isNaN(Number(raw)) &&
      /^-?\d+(\.\d+)?$/.test(raw.trim())
    ) {
      return Number(raw);
    }
    if (
      (raw.startsWith("{") && raw.endsWith("}")) ||
      (raw.startsWith("[") && raw.endsWith("]"))
    ) {
      try {
        return JSON.parse(raw) as unknown;
      } catch {
        return raw;
      }
    }
  }

  return raw;
}

export function documentListTitle(doc: Record<string, unknown>): string {
  const name = doc.name;
  if (typeof name === "string" && name.trim()) {
    return name;
  }
  const title = doc.title;
  if (typeof title === "string" && title.trim()) {
    return title;
  }
  return String(doc.$id ?? "");
}

export function documentListSubtitle(doc: Record<string, unknown>): string {
  const data = stripDocumentSystemFields(doc);
  const preview = Object.entries(data)
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${String(value).slice(0, 40)}`)
    .join(" · ");
  return preview || String(doc.$id ?? "");
}
