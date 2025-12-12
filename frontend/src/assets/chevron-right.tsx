interface ChevronRightProps {
  width?: number;
  height?: number;
  active?: boolean;
}

export function ChevronRight({
  width = 20,
  height = 20,
  active,
}: ChevronRightProps) {
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
        d="M8.79602 4.99634L13.3449 10.0001L8.79602 15.0038L7.87109 14.163L11.6556 10.0001L7.87109 5.83718L8.79602 4.99634Z"
        fill={active ? "#D4D4D4" : "#525252"}
      />
    </svg>
  );
}
