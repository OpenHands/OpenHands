import { FaCircleInfo } from "react-icons/fa6";

interface MicroagentManagementNoRepositoriesProps {
  title: string;
  documentationUrl: string;
}

export function MicroagentManagementNoRepositories({
  title,
  documentationUrl,
}: MicroagentManagementNoRepositoriesProps) {
  return (
    <div className="flex items-center justify-center pt-10">
      <div className="flex items-center gap-2">
        <h2 className="text-content text-sm font-medium">{title}</h2>
        <a href={documentationUrl} target="_blank" rel="noopener noreferrer">
          <FaCircleInfo className="text-primary" />
        </a>
      </div>
    </div>
  );
}
