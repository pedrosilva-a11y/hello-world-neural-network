export function formatPercent(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown";
  }

  return `${(value * 100).toFixed(2)}%`;
}

export function formatPercentPoints(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown";
  }

  return `${value.toFixed(2)} pp`;
}

export function formatLoss(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown";
  }

  return value.toFixed(4);
}

export function formatNumber(value: number | null): string {
  if (value === null || Number.isNaN(value)) {
    return "Unknown";
  }

  return value.toLocaleString(undefined, {
    maximumSignificantDigits: 6,
  });
}

export function formatBoolean(value: boolean | null): string {
  if (value === null) {
    return "Unknown";
  }

  return value ? "Yes" : "No";
}
