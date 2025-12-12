import React from "react";
import { cn } from "#/utils/utils";
import { BrandBadge } from "../badge";
import XIcon from "#/icons/x.svg?react";

interface BadgeInputProps {
  name?: string;
  value: string[];
  placeholder?: string;
  onChange: (value: string[]) => void;
}

export function BadgeInput({
  name,
  value,
  placeholder,
  onChange,
}: BadgeInputProps) {
  const [inputValue, setInputValue] = React.useState("");

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // If pressing Backspace with empty input, remove the last badge
    if (e.key === "Backspace" && inputValue === "" && value.length > 0) {
      const newBadges = [...value];
      newBadges.pop();
      onChange(newBadges);
      return;
    }

    // If pressing Space or Enter with non-empty input, add a new badge
    if (e.key === " " && inputValue.trim() !== "") {
      e.preventDefault();
      const newBadge = inputValue.trim();
      onChange([...value, newBadge]);
      setInputValue("");
    }
  };

  const removeBadge = (indexToRemove: number) => {
    onChange(value.filter((_, index) => index !== indexToRemove));
  };

  return (
    <div
      className={cn(
        "bg-tertiary border border-[#717888] rounded w-full p-2 placeholder:italic placeholder:text-tertiary-alt",
        "flex flex-wrap items-center gap-2",
      )}
    >
      {value.map((badge, index) => (
        <div key={index}>
          <BrandBadge className="flex items-center gap-0.5 py-1 px-2.5 text-sm text-[#0D0F11] font-semibold leading-[16px]">
            {badge}
            <button
              data-testid="remove-button"
              type="button"
              onClick={() => removeBadge(index)}
              className="cursor-pointer"
            >
              <XIcon width={14} height={14} color="#000000" />
            </button>
          </BrandBadge>
        </div>
      ))}
      <input
        data-testid={name || "badge-input"}
        name={name}
        value={inputValue}
        placeholder={value.length === 0 ? placeholder : ""}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-grow outline-none bg-transparent"
      />
    </div>
  );
}
