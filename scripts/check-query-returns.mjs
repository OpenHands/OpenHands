import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  isIdentifier,
  isImportDeclaration,
  isPropertyAssignment,
  isStringLiteral,
} from "typescript/unstable/ast";
import { API, SignatureKind, TypeFlags } from "typescript/unstable/sync";

function includesMissingData(checker, type) {
  if (!type) return false;
  if (type.flags & (TypeFlags.Void | TypeFlags.Undefined)) return true;
  if (type.isUnionType()) {
    return type
      .getTypes()
      .some((member) => includesMissingData(checker, member));
  }
  if (type.isTypeReference()) {
    const name = type.getTarget().getSymbol()?.name;
    if (name === "Promise" || name === "PromiseLike") {
      return includesMissingData(checker, checker.getTypeArguments(type)[0]);
    }
  }
  return false;
}

// The syntax-only Query plugin cannot enforce the previous no-void-query-fn rule.
export function findMissingQueryReturns(config, sourceDirectory) {
  const api = new API();
  try {
    const project = api
      .updateSnapshot({ openProjects: [config] })
      .getProject(config);
    if (!project)
      throw new Error(`Could not open TypeScript project: ${config}`);
    const { checker, program } = project;
    const failures = [];
    for (const file of project.rootFiles) {
      if (
        !file.startsWith(`${sourceDirectory}${path.sep}`) ||
        !/\.[cm]?tsx?$/.test(file)
      )
        continue;
      const source = program.getSourceFile(file);
      if (
        !source?.statements.some(
          (node) =>
            isImportDeclaration(node) &&
            isStringLiteral(node.moduleSpecifier) &&
            node.moduleSpecifier.text === "@tanstack/react-query",
        )
      )
        continue;
      function visit(node) {
        if (
          isPropertyAssignment(node) &&
          isIdentifier(node.name) &&
          node.name.text === "queryFn"
        ) {
          const type = checker.getTypeAtLocation(node.initializer);
          const signature =
            type && checker.getSignaturesOfType(type, SignatureKind.Call)[0];
          if (
            signature &&
            includesMissingData(
              checker,
              checker.getReturnTypeOfSignature(signature),
            )
          ) {
            const line = source.text.slice(0, node.name.pos).split("\n").length;
            failures.push(
              `${path.relative(process.cwd(), file)}:${line}: queryFn must return a non-undefined value`,
            );
          }
        }
        node.forEachChild(visit);
      }
      visit(source);
    }
    return failures;
  } finally {
    api.close();
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  const failures = findMissingQueryReturns(
    path.resolve("tsconfig.json"),
    path.resolve("src"),
  );
  for (const failure of failures) console.error(failure);
  if (failures.length) process.exitCode = 1;
}
