import { describe, expect, it } from "vitest";

import {
  getTrainingCurveExperiments,
  getTrainingCurveSeries,
  getValidationLossMinimumPoint,
} from "./trainingCurveData";
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

describe("training curve data", () => {
  it("sorts experiments by validation accuracy and limits them", () => {
    const experiments = [
      createExperiment({
        experimentName: "low",
        finalValidationAccuracy: 0.8,
      }),
      createExperiment({
        experimentName: "high",
        finalValidationAccuracy: 0.95,
      }),
      createExperiment({
        experimentName: "middle",
        finalValidationAccuracy: 0.9,
      }),
    ];

    expect(
      getTrainingCurveExperiments(experiments, 2).map(
        (item) => item.experimentName,
      ),
    ).toEqual([
      "high",
      "middle",
    ]);
  });

  it("builds accuracy series as percentages", () => {
    const series = getTrainingCurveSeries(
      [
        createExperiment({
          experimentName: "run",
          trainAccuracy: [0.5, 0.75],
          validationAccuracy: [0.4, 0.7],
        }),
      ],
      "accuracy",
      100,
    );

    expect(series).toEqual([
      {
        experimentName: "run",
        label: "run · train",
        kind: "train",
        points: [
          { x: 1, y: 50 },
          { x: 2, y: 75 },
        ],
      },
      {
        experimentName: "run",
        label: "run · validation",
        kind: "validation",
        points: [
          { x: 1, y: 40 },
          { x: 2, y: 70 },
        ],
      },
    ]);
  });

  it("finds the validation-loss minimum point", () => {
    const point = getValidationLossMinimumPoint(
      createExperiment({
        validationLoss: [0.5, 0.2, 0.3],
      }),
    );

    expect(point).toEqual({
      x: 2,
      y: 0.2,
    });
  });

  it("returns null when there is no validation loss", () => {
    expect(getValidationLossMinimumPoint(createExperiment({}))).toBeNull();
  });
});

describe("training curve max iteration filtering", () => {
  it("limits curve points by max iteration before downsampling", () => {
    const series = getTrainingCurveSeries(
      [
        createExperiment({
          experimentName: "run",
          trainLoss: [0.5, 0.4, 0.3, 0.2],
        }),
      ],
      "loss",
      100,
      10,
      2,
    );

    expect(series[0].points).toEqual([
      { x: 1, y: 0.5 },
      { x: 2, y: 0.4 },
    ]);
  });

  it("finds the validation-loss minimum only inside the max iteration range", () => {
    const point = getValidationLossMinimumPoint(
      createExperiment({
        validationLoss: [0.5, 0.4, 0.1],
      }),
      2,
    );

    expect(point).toEqual({
      x: 2,
      y: 0.4,
    });
  });
});

describe("training curve line visibility", () => {
  it("can return only training curves", () => {
    const series = getTrainingCurveSeries(
      [
        createExperiment({
          experimentName: "run",
          trainLoss: [0.5, 0.4],
          validationLoss: [0.6, 0.45],
        }),
      ],
      "loss",
      100,
      10,
      null,
      ["train"],
    );

    expect(series).toHaveLength(1);
    expect(series[0].kind).toBe("train");
  });

  it("can return only validation curves", () => {
    const series = getTrainingCurveSeries(
      [
        createExperiment({
          experimentName: "run",
          trainLoss: [0.5, 0.4],
          validationLoss: [0.6, 0.45],
        }),
      ],
      "loss",
      100,
      10,
      null,
      ["validation"],
    );

    expect(series).toHaveLength(1);
    expect(series[0].kind).toBe("validation");
  });
});