interface ChevronLeftProps {
  width?: number;
  height?: number;
  active?: boolean;
}

export function ChevronLeft({
  width = 20,
  height = 20,
  active,
}: ChevronLeftProps) {
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M11.204 15.0037L6.65511 9.99993L11.204 4.99617L12.1289 5.83701L8.34444 9.99993L12.1289 14.1628L11.204 15.0037Z"
        fill={active ? "#D4D4D4" : "#525252"}
      />
    </svg>
  );
}
