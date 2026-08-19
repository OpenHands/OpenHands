import {
  builtinEnvironments,
  type Environment,
  type EnvironmentReturn,
} from "vitest/runtime";

class ProgressEventFallback extends Event implements ProgressEvent {
  readonly lengthComputable: boolean;
  readonly loaded: number;
  readonly total: number;

  constructor(type: string, init: ProgressEventInit = {}) {
    super(type, init);
    this.lengthComputable = init.lengthComputable ?? false;
    this.loaded = init.loaded ?? 0;
    this.total = init.total ?? 0;
  }
}

Reflect.deleteProperty(globalThis, "localStorage");

const environment: Environment = {
  ...builtinEnvironments.jsdom,
  name: "jsdom-with-progress-event-fallback",
  async setup(global, options): Promise<EnvironmentReturn> {
    const environmentReturn = await builtinEnvironments.jsdom.setup(
      global,
      options,
    );

    return {
      ...environmentReturn,
      async teardown(teardownGlobal) {
        await environmentReturn.teardown(teardownGlobal);
        Object.defineProperty(teardownGlobal, "ProgressEvent", {
          configurable: true,
          writable: true,
          value: ProgressEventFallback,
        });
      },
    };
  },
};

export default environment;
