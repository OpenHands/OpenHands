import { runInContext } from "node:vm";
import { builtinEnvironments, type Environment } from "vitest/runtime";

export default {
  name: "canvas-jsdom",
  viteEnvironment: "client",
  setup: builtinEnvironments.jsdom.setup,
  async setupVM(options) {
    const vm = await builtinEnvironments.node.setupVM!(options);
    const context = vm.getVmContext();
    const global = runInContext("globalThis", context);
    // Use the same configurable browser globals as ordinary jsdom workers.
    const dom = await builtinEnvironments.jsdom.setup(global, options);
    return {
      getVmContext: () => context,
      async teardown() {
        try {
          await dom.teardown(global);
        } finally {
          await vm.teardown();
        }
      },
    };
  },
} satisfies Environment;
