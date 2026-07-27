import React from "react";
import { ExtraProps } from "react-markdown";

// Custom component to render <ul> in markdown
export function ul({
  children,
}: React.ClassAttributes<HTMLUListElement> &
  React.HTMLAttributes<HTMLUListElement> &
  ExtraProps) {
  return (
    <ul className="my-2 list-disc ml-5 pl-2 whitespace-normal leading-6">
      {children}
    </ul>
  );
}

// Custom component to render <ol> in markdown
export function ol({
  children,
  start,
}: React.ClassAttributes<HTMLOListElement> &
  React.OlHTMLAttributes<HTMLOListElement> &
  ExtraProps) {
  return (
    <ol
      className="my-2 list-decimal ml-5 pl-2 whitespace-normal leading-6"
      start={start}
    >
      {children}
    </ol>
  );
}

// Custom component to render <li> in markdown
export function li({
  children,
}: React.ClassAttributes<HTMLLIElement> &
  React.LiHTMLAttributes<HTMLLIElement> &
  ExtraProps) {
  return <li className="py-0.5">{children}</li>;
}
