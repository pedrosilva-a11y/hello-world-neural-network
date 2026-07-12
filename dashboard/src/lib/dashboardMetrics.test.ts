import { describe, expect, it } from "vitest";

import { getDashboardMetrics } from "./dashboardMetrics";
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

describe("getDashboardMetrics", () => {
  it("returns empty metrics when there are no experiments", () => {
    const metrics = getDashboardMetrics([]);

    expect(metrics.experimentCount).toBe(0);
    expect(metrics.bestValidationAccuracy).toBeNull();
    expect(metrics.bestValidationAccuracyExperimentName).toBeNull();
    expect(metrics.averageValidationAccuracy).toBeNull();
  });

  it("computes best and average validation metrics", () => {
    const metrics = getDashboardMetrics([
      createExperiment({
        experimentName: "baseline",
        finalValidationAccuracy: 0.91,
        finalValidationLoss: 0.25,
        validationErrorPercent: 9,
      }),
      createExperiment({
        experimentName: "best_accuracy",
        finalValidationAccuracy: 0.97,
        finalValidationLoss: 0.14,
        validationErrorPercent: 3,
      }),
      createExperiment({
        experimentName: "best_loss",
        finalValidationAccuracy: 0.96,
        finalValidationLoss: 0.1,
        validationErrorPercent: 4,
      }),
    ]);

    expect(metrics.experimentCount).toBe(3);
    expect(metrics.bestValidationAccuracy).toBe(0.97);
    expect(metrics.bestValidationAccuracyExperimentName).toBe("best_accuracy");
    expect(metrics.bestValidationLoss).toBe(0.1);
    expect(metrics.bestValidationLossExperimentName).toBe("best_loss");
    expect(metrics.lowestValidationErrorPercent).toBe(3);
    expect(metrics.lowestValidationErrorExperimentName).toBe("best_accuracy");
    expect(metrics.averageValidationAccuracy).toBeCloseTo(0.9466666667);
  });

  it("ignores unknown metric values", () => {
    const metrics = getDashboardMetrics([
      createExperiment({
        experimentName: "missing",
      }),
      createExperiment({
        experimentName: "known",
        finalValidationAccuracy: 0.8,
        finalValidationLoss: 0.4,
        validationErrorPercent: 20,
      }),
    ]);

    expect(metrics.bestValidationAccuracy).toBe(0.8);
    expect(metrics.bestValidationAccuracyExperimentName).toBe("known");
    expect(metrics.averageValidationAccuracy).toBe(0.8);
  });
});
