import React from "react";
import { ExtraProps } from "react-markdown";

// Custom component to render <h1> in markdown
export function h1({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h1 className="text-xl text-white font-medium leading-7 mb-3 mt-4 first:mt-0">
      {children}
    </h1>
  );
}

// Custom component to render <h2> in markdown
export function h2({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h2 className="text-lg font-medium leading-6 -tracking-[0.01em] text-white mb-2.5 mt-4 first:mt-0">
      {children}
    </h2>
  );
}

// Custom component to render <h3> in markdown
export function h3({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h3 className="text-base font-medium text-white mb-2 mt-3 first:mt-0">
      {children}
    </h3>
  );
}

// Custom component to render <h4> in markdown
export function h4({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h4 className="text-sm font-medium text-white mb-1.5 mt-3 first:mt-0">
      {children}
    </h4>
  );
}

// Custom component to render <h5> in markdown
export function h5({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h5 className="text-sm font-normal text-[var(--oh-text-tertiary)] mb-1.5 mt-2.5 first:mt-0">
      {children}
    </h5>
  );
}

// Custom component to render <h6> in markdown
export function h6({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h6 className="text-sm font-normal text-[var(--oh-text-tertiary)] mb-1.5 mt-2.5 first:mt-0">
      {children}
    </h6>
  );
}
