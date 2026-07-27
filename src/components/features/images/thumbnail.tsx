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
        size === "small" && "w-[62px] h-[62px]",
        size === "large" && "w-[100px] h-[100px]",
      )}
    />
  );
}
