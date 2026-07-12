import { describe, expect, it } from "vitest";

import { downsampleSeries } from "./downsample";

describe("downsampleSeries", () => {
  it("returns all points when the series is already small enough", () => {
    const points = [
      { x: 1, y: 10 },
      { x: 2, y: 20 },
    ];

    expect(downsampleSeries(points, 5)).toEqual(points);
  });

  it("preserves the first and last points", () => {
    const points = Array.from({ length: 10 }, (_, index) => ({
      x: index + 1,
      y: index,
    }));

    const downsampled = downsampleSeries(points, 4);

    expect(downsampled).toHaveLength(4);
    expect(downsampled[0]).toEqual({ x: 1, y: 0 });
    expect(downsampled[downsampled.length - 1]).toEqual({ x: 10, y: 9 });
  });

  it("returns an empty array when maxPoints is zero", () => {
    expect(downsampleSeries([{ x: 1, y: 1 }], 0)).toEqual([]);
  });
});
