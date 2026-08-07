import { describe, expect, it } from "vitest";
import type { AppwriteAttribute } from "#/api/integrations/appwrite-service";
import {
  buildDocumentFormFields,
  buildDocumentFormValues,
  documentListTitle,
  formValuesToDocumentData,
} from "#/components/features/cloudai/cloudai-document-form";

const labels = {
  id: "ID",
  arrayHelp: "array help",
  datetimeHelp: "datetime help",
  jsonFallback: "Data (JSON)",
};

describe("cloudai-document-form", () => {
  it("builds one labeled field per attribute", () => {
    const attributes: AppwriteAttribute[] = [
      { key: "name", type: "string", required: true },
      { key: "cnpj", type: "string" },
      { key: "active", type: "boolean" },
      { key: "tags", type: "string", array: true },
    ];

    const fields = buildDocumentFormFields(
      attributes,
      undefined,
      labels,
      true,
    );

    expect(fields.map((f) => f.key)).toEqual([
      "id",
      "name",
      "cnpj",
      "active",
      "tags",
    ]);
    expect(fields.find((f) => f.key === "active")?.checkbox).toBe(true);
    expect(fields.find((f) => f.key === "tags")?.multiline).toBe(true);
    expect(fields.find((f) => f.key === "name")?.required).toBe(true);
  });

  it("falls back to document keys when attributes are empty", () => {
    const fields = buildDocumentFormFields(
      [],
      {
        $id: "doc-1",
        name: "ASSOCIACAO",
        cnpj: "11.050.047/0001-86",
        logo_url: "https://example.com/logo.png",
      },
      labels,
      false,
    );

    expect(fields.map((f) => f.key)).toEqual(["name", "cnpj", "logo_url"]);
    expect(fields.every((f) => f.label === f.key)).toBe(true);
  });

  it("round-trips typed form values into document data", () => {
    const attributes: AppwriteAttribute[] = [
      { key: "name", type: "string" },
      { key: "count", type: "integer" },
      { key: "active", type: "boolean" },
      { key: "tags", type: "string", array: true },
    ];

    const values = buildDocumentFormValues(
      attributes,
      {
        name: "Assoc",
        count: 3,
        active: true,
        tags: ["a", "b"],
      },
      false,
    );

    expect(values).toEqual({
      name: "Assoc",
      count: "3",
      active: "true",
      tags: '["a","b"]',
    });

    expect(formValuesToDocumentData(values, attributes)).toEqual({
      name: "Assoc",
      count: 3,
      active: true,
      tags: ["a", "b"],
    });
  });

  it("prefers name for document list title", () => {
    expect(
      documentListTitle({
        $id: "827ea16b-75e6-45b0-94db-0011a086aa4a",
        name: "ASSOCIACAO BRASILEIRA DE EVENTOS",
      }),
    ).toBe("ASSOCIACAO BRASILEIRA DE EVENTOS");
  });
});
