import { describe, expect, it } from "vitest";

import {
  getComparisonChartPoints,
  type ComparisonMetric,
} from "./comparisonChartData";
import type { NormalizedExperiment } from "../types/summary";

function createExperiment(
  overrides: Partial<NormalizedExperiment>,
): NormalizedExperiment {
  return {
    experimentName: "experiment",
    modelName: "model",
    neuronsProfile: [],
    neuronsProfileLabel: "Unknown",
    normalizePixels: null,
    optimizer: "unknown",
    learningRate: null,
    learningRateKey: "unknown",
    numIterations: null,
    numIterationsKey: "unknown",
    trainLoss: [],
    validationLoss: [],
    trainAccuracy: [],
    validationAccuracy: [],
    finalTrainLoss: null,
    finalValidationLoss: null,
    finalTrainAccuracy: null,
    finalValidationAccuracy: null,
    validationErrorPercent: null,
    trainValidationGapPercent: null,
    modifiedAt: null,
    raw: {},
    ...overrides,
  };
}

describe("getComparisonChartPoints", () => {
  it("sorts validation accuracy descending and converts to percent", () => {
    const points = getComparisonChartPoints(
      [
        createExperiment({
          experimentName: "baseline",
          finalValidationAccuracy: 0.91,
        }),
        createExperiment({
          experimentName: "best",
          finalValidationAccuracy: 0.97,
        }),
      ],
      "validationAccuracy",
    );

    expect(points).toEqual([
      {
        experimentName: "best",
        value: 97,
        formattedValue: "97.00%",
      },
      {
        experimentName: "baseline",
        value: 91,
        formattedValue: "91.00%",
      },
    ]);
  });

  it("sorts validation loss ascending and ignores unknown values", () => {
    const points = getComparisonChartPoints(
      [
        createExperiment({
          experimentName: "missing",
        }),
        createExperiment({
          experimentName: "higher_loss",
          finalValidationLoss: 0.4,
        }),
        createExperiment({
          experimentName: "lower_loss",
          finalValidationLoss: 0.1,
        }),
      ],
      "validationLoss",
    );

    expect(points.map((point) => point.experimentName)).toEqual([
      "lower_loss",
      "higher_loss",
    ]);
  });

  it("limits the number of chart points", () => {
    const experiments = Array.from({ length: 5 }, (_, index) =>
      createExperiment({
        experimentName: `experiment_${index}`,
        finalValidationAccuracy: 0.8 + index / 100,
      }),
    );

    const points = getComparisonChartPoints(
      experiments,
      "validationAccuracy",
      3,
    );

    expect(points).toHaveLength(3);
    expect(points[0].experimentName).toBe("experiment_4");
  });

  it.each<ComparisonMetric>([
    "validationAccuracy",
    "validationLoss",
    "validationErrorPercent",
    "trainValidationGapPercent",
  ])("returns no points when %s has no known values", (metric) => {
    expect(getComparisonChartPoints([createExperiment({})], metric)).toEqual([]);
  });
});
