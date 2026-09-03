export function activate(host) {
  return host.registerPage("hello", ({ container, path }) => {
    const wrapper = document.createElement("section");
    wrapper.style.cssText =
      "min-height:100%;padding:2rem;color:var(--oh-text-primary,#fff);background:var(--oh-color-base,#111);";

    const title = document.createElement("h1");
    title.textContent = "Hello from a Canvas Extension";
    title.style.cssText = "font-size:1.5rem;font-weight:600;";

    const detail = document.createElement("p");
    detail.textContent = path
      ? `Nested extension path: ${path}`
      : `Host API ${host.apiVersion} on backend ${host.backend.id}`;
    detail.style.cssText = "margin-top:0.75rem;opacity:0.75;";

    wrapper.append(title, detail);
    container.append(wrapper);
    return () => wrapper.remove();
  });
}
