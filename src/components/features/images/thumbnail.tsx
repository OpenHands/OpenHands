import { cn } from "#/utils/utils";

interface ThumbnailProps {
  src: string;
  size?: "small" | "large";
}

export function Thumbnail({ src, size = "small" }: ThumbnailProps) {
  return (
    <img
      role="img"
      alt=""
      src={src}
      className={cn(
        "rounded-sm object-cover",
        size === "small" && "w-15.5 h-15.5",
        size === "large" && "w-25 h-25",
      )}
    />
  );
}
