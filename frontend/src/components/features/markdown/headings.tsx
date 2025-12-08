import React from "react";
import { ExtraProps } from "react-markdown";

// Custom component to render <h1> in markdown
export function h1({
  children,
}: React.ClassAttributes<HTMLHeadingElement> &
  React.HTMLAttributes<HTMLHeadingElement> &
  ExtraProps) {
  return (
    <h1 className="text-2xl text-white font-bold leading-8 mb-4 mt-6 first:mt-0">
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
    <h2 className="text-xl font-semibold leading-6 -tracking-[0.02em] text-white mb-3 mt-5 first:mt-0">
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
    <h3 className="text-lg font-semibold text-white mb-2 mt-4 first:mt-0">
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
    <h4 className="text-base font-semibold text-white mb-2 mt-4 first:mt-0">
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
    <h5 className="text-sm font-semibold text-white mb-2 mt-3 first:mt-0">
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
    <h6 className="text-sm font-medium text-gray-300 mb-2 mt-3 first:mt-0">
      {children}
    </h6>
  );
}
