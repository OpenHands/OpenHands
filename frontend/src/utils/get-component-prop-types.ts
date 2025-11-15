export type GetComponentPropTypes<T> =
  T extends React.ComponentType<infer P> ? P : never;
