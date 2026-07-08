/* eslint-disable i18next/no-literal-string */
import React from "react";
import {
  TrendDownIcon,
  TrendUpIcon,
} from "#/components/shared/icons/inline-icons";
import { formatShortDate } from "./usage-dashboard-utils";

export function KPICard({
  label,
  value,
  trend,
  trendUp,
}: {
  label: string;
  value: string | number;
  trend?: string;
  trendUp?: boolean;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-5">
      <span className="text-zinc-500 text-xs font-medium uppercase tracking-wide">
        {label}
      </span>
      <div className="text-white text-2xl font-bold mt-2">{value}</div>
      {trend && (
        <div
          className={`flex items-center gap-1 mt-2 text-xs ${trendUp ? "text-green-400" : "text-red-400"}`}
        >
          {trendUp ? <TrendUpIcon /> : <TrendDownIcon />}
          {trend}
        </div>
      )}
    </div>
  );
}

export function AreaChart({
  data,
}: {
  data: { date: string; value: number }[];
}) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const minValue = Math.min(...data.map((d) => d.value), 0);
  const range = maxValue - minValue || 1;

  const width = 100;
  const height = 100;
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((d.value - minValue) / range) * height;
    return `${x},${y}`;
  });

  const pathD = `M ${points.join(" L ")}`;
  const areaD = `${pathD} L ${width},${height} L 0,${height} Z`;

  return (
    <div className="relative h-48 w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full"
        preserveAspectRatio="none"
      >
        {[0, 25, 50, 75, 100].map((pct) => (
          <line
            key={pct}
            x1="0"
            y1={`${pct}%`}
            x2="100%"
            y2={`${pct}%`}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="0.5"
          />
        ))}
        <path d={areaD} fill="url(#blueGradient)" opacity="0.3" />
        <path d={pathD} fill="none" stroke="#3B82F6" strokeWidth="2" />
        <defs>
          <linearGradient id="blueGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-zinc-600 -ml-2">
        <span>{maxValue.toLocaleString()}</span>
        <span>{Math.round((maxValue + minValue) / 2).toLocaleString()}</span>
        <span>{minValue.toLocaleString()}</span>
      </div>
      <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-zinc-600 mt-2">
        {data
          .filter((_, i) => i % Math.ceil(data.length / 7) === 0)
          .map((d) => (
            <span key={d.date}>{formatShortDate(d.date)}</span>
          ))}
      </div>
    </div>
  );
}

export function PieChart({
  data,
  total,
}: {
  data: { value: number; color: string }[];
  total: number;
}) {
  const size = 160;
  const strokeWidth = 18;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <div className="relative h-40 w-40">
      <svg viewBox={`0 0 ${size} ${size}`} className="h-full w-full">
        <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={strokeWidth}
          />
          {data.map((segment, index) => {
            const portion = total > 0 ? segment.value / total : 0;
            const segmentLength = circumference * portion;
            const dashArray = `${segmentLength} ${circumference - segmentLength}`;
            const segmentOffset = offset;
            offset += segmentLength;
            return (
              <circle
                key={`${segment.color}-${index}`}
                cx={size / 2}
                cy={size / 2}
                r={radius}
                fill="transparent"
                stroke={segment.color}
                strokeWidth={strokeWidth}
                strokeDasharray={dashArray}
                strokeDashoffset={-segmentOffset}
                strokeLinecap="round"
              />
            );
          })}
        </g>
      </svg>
    </div>
  );
}
