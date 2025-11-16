import React, { useRef, useEffect } from "react";

interface FileAutocompleteProps {
  isOpen: boolean;
  files: string[];
  selectedIndex: number;
  position: { top: number; left: number };
  showAbove: boolean;
  onSelect: (file: string) => void;
  onClose: () => void;
}

/**
 * Extract filename from path
 */
const getFileName = (path: string): string => {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
};

/**
 * Extract directory from path
 */
const getDirectory = (path: string): string => {
  const parts = path.split("/");
  if (parts.length <= 1) return "/";
  return parts.slice(0, -1).join("/");
};

/**
 * Autocomplete dropdown for file selection
 * Shows filtered list of files with keyboard navigation support
 */
export function FileAutocomplete({
  isOpen,
  files,
  selectedIndex,
  position,
  showAbove,
  onSelect,
  onClose,
}: FileAutocompleteProps) {
  const dropdownRef = useRef<HTMLDivElement>(null);
  const selectedItemRef = useRef<HTMLDivElement>(null);

  // Scroll selected item into view
  useEffect(() => {
    if (selectedItemRef.current) {
      selectedItemRef.current.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [selectedIndex]);

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
    return undefined;
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const style: React.CSSProperties = {
    position: "fixed",
    left: `${position.left}px`,
    [showAbove ? "bottom" : "top"]: showAbove
      ? `${window.innerHeight - position.top + 5}px`
      : `${position.top + 5}px`,
    zIndex: 9999,
  };

  return (
    <div
      ref={dropdownRef}
      style={style}
      className="bg-[#1e1f26] border border-[#3b3d45] rounded-lg shadow-lg max-h-[250px] overflow-y-auto min-w-[300px] max-w-[500px] custom-scrollbar"
    >
      {files.length === 0 ? (
        // eslint-disable-next-line i18next/no-literal-string
        <div className="px-4 py-3 text-gray-400 text-sm">No files found</div>
      ) : (
        <div className="py-1">
          {files.map((file, index) => (
            <div
              key={file}
              ref={index === selectedIndex ? selectedItemRef : null}
              className={`px-4 py-2 cursor-pointer transition-colors ${
                index === selectedIndex
                  ? "bg-[#2a2d35] border-l-2 border-blue-500"
                  : "hover:bg-[#25272d]"
              }`}
              onClick={() => onSelect(file)}
            >
              <div className="text-white text-sm font-medium truncate">
                {getFileName(file)}
              </div>
              <div className="text-gray-400 text-xs truncate">
                {getDirectory(file)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
