import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "#/utils/utils";

const typographyVariants = cva("", {
  variants: {
    variant: {
      h1: "text-[32px] text-white font-bold leading-5",
      h2: "text-xl font-semibold leading-6 -tracking-[0.02em] text-white",
      h3: "text-sm font-semibold text-gray-300",
      span: "text-sm font-normal text-white leading-5.5",
      p: "text-sm font-normal text-white leading-5.5",
      codeBlock:
        "font-mono text-sm leading-relaxed text-gray-300 whitespace-pre-wrap",
    },
  },
  defaultVariants: {
    variant: "h1",
  },
});

interface TypographyProps extends VariantProps<typeof typographyVariants> {
  className?: string;
  testId?: string;
  children: React.ReactNode;
}

export function Typography({
  variant,
  className,
  testId,
  children,
}: TypographyProps) {
  const Tag = variant as keyof React.JSX.IntrinsicElements;

  return (
    <Tag
      data-testid={testId}
      className={cn(typographyVariants({ variant }), className)}
    >
      {children}
    </Tag>
  );
}

// Export individual heading components for convenience
export function H1({
  className,
  testId,
  children,
}: Omit<TypographyProps, "variant">) {
  return (
    <Typography variant="h1" className={className} testId={testId}>
      {children}
    </Typography>
  );
}

export function H2({
  className,
  testId,
  children,
}: Omit<TypographyProps, "variant">) {
  return (
    <Typography variant="h2" className={className} testId={testId}>
      {children}
    </Typography>
  );
}

export function H3({
  className,
  testId,
  children,
}: Omit<TypographyProps, "variant">) {
  return (
    <Typography variant="h3" className={className} testId={testId}>
      {children}
    </Typography>
  );
}

export function Text({
  className,
  testId,
  children,
}: Omit<TypographyProps, "variant">) {
  return (
    <Typography variant="span" className={className} testId={testId}>
      {children}
    </Typography>
  );
}

export function CodeBlock({
  className,
  testId,
  children,
}: Omit<TypographyProps, "variant">) {
  return (
    <Typography variant="codeBlock" className={className} testId={testId}>
      {children}
    </Typography>
  );
}

export function Paragraph({
  className,
  testId,
  children,
}: Omit<TypographyProps, "variant">) {
  return (
    <Typography variant="p" className={className} testId={testId}>
      {children}
    </Typography>
  );
}

// Attach components to Typography for the expected API
Typography.H1 = H1;
Typography.H2 = H2;
Typography.H3 = H3;
Typography.Text = Text;
Typography.CodeBlock = CodeBlock;
Typography.Paragraph = Paragraph;
