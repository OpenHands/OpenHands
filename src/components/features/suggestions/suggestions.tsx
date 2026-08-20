import { SuggestionItem, type Suggestion } from "./suggestion-item";

interface SuggestionsProps {
  suggestions: Suggestion[];
  onSuggestionClick: (value: string) => void;
}

export function Suggestions({
  suggestions,
  onSuggestionClick,
}: SuggestionsProps) {
  return (
    <div data-testid="suggestions" className="grid grid-cols-2 gap-5 max-w-fit">
      {suggestions.map((suggestion, index) => (
        <SuggestionItem
          key={index}
          suggestion={suggestion}
          onClick={onSuggestionClick}
        />
      ))}
    </div>
  );
}
