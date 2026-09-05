const SVG_EDIT_VERSION = "7.4.2";
const SVG_EDIT_URL = `https://unpkg.com/svgedit@${SVG_EDIT_VERSION}/dist/editor/index.html`;

export { SVG_EDIT_VERSION, SVG_EDIT_URL };

export function activate(host) {
  return host.registerPage("editor", ({ container }) => {
    const frame = document.createElement("iframe");
    frame.title = "SVG Editor";
    frame.src = SVG_EDIT_URL;
    frame.referrerPolicy = "no-referrer";
    frame.allow = "clipboard-read; clipboard-write";
    frame.style.cssText =
      "display:block;width:100%;height:100%;border:0;background:#fff;";
    frame.dataset.canvasExtension = host.extension.name;
    container.append(frame);

    return () => frame.remove();
  });
}
