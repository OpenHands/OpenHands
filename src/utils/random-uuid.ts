// eslint-disable-next-line no-restricted-imports -- the one sanctioned v4 call site
import { v4 as uuidv4 } from "uuid";

/**
 * Random UUIDv4 that also works outside secure contexts.
 *
 * `crypto.randomUUID` is SecureContext-only, so it is undefined when Canvas
 * is opened over plain HTTP on a non-loopback host (#19514). `uuid` calls it
 * when present and otherwise builds a v4 from `crypto.getRandomValues`, which
 * every context exposes.
 */
export const randomUUID = (): string => uuidv4();
