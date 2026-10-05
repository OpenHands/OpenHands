// The environment for (re)starting a run's launcher. Run-scoped values
// (ports, state paths, private HOME) come from the environment saved at
// launch. Machine-scoped values (PATH, proxy, CA bundles: the pass-through
// names) come from the current shell, because they can change between
// sessions: a restart that replayed a stale proxy port could no longer reach
// the model provider.
export function launcherEnvFor(saved, current, passNames) {
  const env = { ...saved };
  for (const name of passNames) {
    if (current[name] !== undefined) env[name] = current[name];
    else delete env[name];
  }
  return env;
}
