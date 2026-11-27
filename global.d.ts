interface Window {
  __GITHUB_CLIENT_ID__?: string | null;
}

declare module "postcss-prefix-selector" {
  interface PrefixerOptions {
    prefix: string;
    transform?: (
      prefix: string,
      selector: string,
      prefixedSelector: string,
    ) => string;
  }

  export default function prefixer(
    options: PrefixerOptions,
  ): import("postcss").AcceptedPlugin;
}
