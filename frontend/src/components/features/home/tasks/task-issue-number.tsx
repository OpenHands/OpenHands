interface TaskIssueNumberProps {
  issueNumber: number;
  href: string;
}

export function TaskIssueNumber({ href, issueNumber }: TaskIssueNumberProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      data-testid="task-id"
    >
      <span className="text-xs text-[#A3A3A3] leading-4 font-normal">
        #{issueNumber}
      </span>
    </a>
  );
}
