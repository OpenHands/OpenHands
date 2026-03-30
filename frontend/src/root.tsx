import {
  Links,
  LinksFunction,
  Meta,
  MetaFunction,
  Outlet,
  Scripts,
  ScrollRestoration,
} from "react-router";
import "./tailwind.css";
import "./index.css";
import React from "react";
import { Toaster } from "react-hot-toast";

function getThemeFromUrl(): "dark" | "light" {
  if (typeof window === "undefined") return "dark";
  const params = new URLSearchParams(window.location.search);
  const theme = params.get("colorScheme");
  return theme === "light" ? "light" : "dark";
}

export function Layout({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = React.useState<"dark" | "light">(getThemeFromUrl);

  React.useEffect(() => {
    const handlePopState = () => setTheme(getThemeFromUrl());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  return (
    <html lang="en" className={theme} data-theme={theme}>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
        <Toaster />
        <div id="modal-portal-exit" />
      </body>
    </html>
  );
}

export const links: LinksFunction = () => [
  { rel: "icon", type: "image/svg+xml", href: "/favicon.ico" },
];

export const meta: MetaFunction = () => [
  { title: "Build.One" },
  { name: "description", content: "Let's Start Building!" },
];

export function HydrateFallback() {
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-base">
      <div className="loader h-2.5 w-2.5 rounded-full" />
    </div>
  );
}

export default function App() {
  return <Outlet />;
}
