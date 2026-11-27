interface SectionCardProps {
  icon: React.ReactNode;
  title: string;
  /** Right-aligned header content, such as the trigger or action kind. */
  meta?: React.ReactNode;
  testId?: string;
  children: React.ReactNode;
}

export function SectionCard({
  icon,
  title,
  meta,
  testId,
  children,
}: SectionCardProps) {
  return (
    <section
      data-testid={testId}
      className="rounded-2xl border border-border bg-surface"
    >
      <div className="flex items-center gap-2 border-b border-border px-5 py-3">
        <span className="size-4 text-muted">{icon}</span>
        <h3 className="text-sm font-medium text-content">{title}</h3>
        {meta ? (
          <span className="ml-auto text-xs text-muted">{meta}</span>
        ) : null}
      </div>
      <div className="px-5 py-5">{children}</div>
    </section>
  );
}
