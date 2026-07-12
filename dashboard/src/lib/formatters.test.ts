import { describe, expect, it } from "vitest";

import {
  formatBoolean,
  formatLoss,
  formatNumber,
  formatPercent,
  formatPercentPoints,
} from "./formatters";

describe("formatters", () => {
  it("formats percentages", () => {
    expect(formatPercent(0.9708)).toBe("97.08%");
    expect(formatPercent(null)).toBe("Unknown");
  });

  it("formats percentage points", () => {
    expect(formatPercentPoints(2.345)).toBe("2.35 pp");
    expect(formatPercentPoints(null)).toBe("Unknown");
  });

  it("formats losses", () => {
    expect(formatLoss(0.106168)).toBe("0.1062");
    expect(formatLoss(null)).toBe("Unknown");
  });

  it("formats numbers", () => {
    expect(formatNumber(5000)).toBe("5,000");
    expect(formatNumber(null)).toBe("Unknown");
  });

  it("formats booleans", () => {
    expect(formatBoolean(true)).toBe("Yes");
    expect(formatBoolean(false)).toBe("No");
    expect(formatBoolean(null)).toBe("Unknown");
  });
});
