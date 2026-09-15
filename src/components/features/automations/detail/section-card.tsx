interface SectionCardProps {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}

export function SectionCard({ icon, title, children }: SectionCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-surface">
      <div className="flex items-center gap-2 border-b border-border px-5 py-3">
        <span className="size-4 text-muted">{icon}</span>
        <h3 className="text-sm font-medium text-content">{title}</h3>
      </div>
      <div className="px-5 py-5">{children}</div>
    </div>
  );
}
