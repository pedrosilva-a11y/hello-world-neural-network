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

    batchingStrategy: "full_batch",
    batchingLabel: "Full batch",
    batchSize: null,
    batchSizeKey: "unknown",
    shuffleBatches: false,
    batchRandomSeed: 42,
    numEpochs: null,
    numEpochsKey: "unknown",

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
    expect(metrics.bestValidationLoss).toBeNull();
    expect(metrics.bestValidationLossExperimentName).toBeNull();
    expect(metrics.lowestValidationErrorPercent).toBeNull();
    expect(metrics.lowestValidationErrorExperimentName).toBeNull();
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
    expect(metrics.bestValidationLoss).toBe(0.4);
    expect(metrics.bestValidationLossExperimentName).toBe("known");
    expect(metrics.lowestValidationErrorPercent).toBe(20);
    expect(metrics.lowestValidationErrorExperimentName).toBe("known");
    expect(metrics.averageValidationAccuracy).toBe(0.8);
  });

  it("keeps dashboard metrics independent from batching metadata", () => {
    const metrics = getDashboardMetrics([
      createExperiment({
        experimentName: "full_batch",
        batchingStrategy: "full_batch",
        batchingLabel: "Full batch",
        batchSize: null,
        batchSizeKey: "unknown",
        numEpochs: null,
        numEpochsKey: "unknown",
        finalValidationAccuracy: 0.93,
        finalValidationLoss: 0.2,
        validationErrorPercent: 7,
      }),
      createExperiment({
        experimentName: "mini_batch",
        batchingStrategy: "mini_batch",
        batchingLabel: "Mini-batch",
        batchSize: 128,
        batchSizeKey: "128",
        shuffleBatches: true,
        batchRandomSeed: 42,
        numEpochs: 20,
        numEpochsKey: "20",
        finalValidationAccuracy: 0.95,
        finalValidationLoss: 0.15,
        validationErrorPercent: 5,
      }),
    ]);

    expect(metrics.experimentCount).toBe(2);
    expect(metrics.bestValidationAccuracy).toBe(0.95);
    expect(metrics.bestValidationAccuracyExperimentName).toBe("mini_batch");
    expect(metrics.bestValidationLoss).toBe(0.15);
    expect(metrics.bestValidationLossExperimentName).toBe("mini_batch");
    expect(metrics.lowestValidationErrorPercent).toBe(5);
    expect(metrics.lowestValidationErrorExperimentName).toBe("mini_batch");
    expect(metrics.averageValidationAccuracy).toBeCloseTo(0.94);
  });
});