import { describe, expect, it } from "vitest";

import { normalizeSummary } from "./normalizeSummary";
import type { RawExperimentSummary } from "../types/summary";

describe("normalizeSummary", () => {
  it("normalizes a current MLP summary", () => {
    const summary: RawExperimentSummary = {
      experiment_name: "relu_128_128_normalized_lr_01_5k",
      config: {
        model: {
          name: "multi_layer_relu_classifier",
          neurons_profile: [128, 128, 10],
        },
        preprocessing: {
          normalize_pixels: true,
        },
        training: {
          learning_rate: 0.1,
          num_iterations: 5000,
          optimizer: "gradient_descent",
          regularization: {
            enabled: true,
            type: "l2",
            lambda: 0.01,
          },
        },
      },
      metrics: {
        train_loss: [0.2, 0.018853909846237197],
        validation_loss: [0.3, 0.1061686398068565],
        train_accuracy: [0.9, 0.997440552364513],
        validation_accuracy: [0.92, 0.9708298606977022],
      },
    };

    const normalized = normalizeSummary(summary);

    expect(normalized.experimentName).toBe("relu_128_128_normalized_lr_01_5k");
    expect(normalized.modelName).toBe("multi_layer_relu_classifier");
    expect(normalized.neuronsProfile).toEqual([128, 128, 10]);
    expect(normalized.neuronsProfileLabel).toBe("[128, 128, 10]");
    expect(normalized.normalizePixels).toBe(true);
    expect(normalized.learningRateKey).toBe("0.1");
    expect(normalized.numIterationsKey).toBe("5000");
    expect(normalized.finalValidationAccuracy).toBe(0.9708298606977022);
    expect(normalized.validationErrorPercent).toBeCloseTo(2.917013930229781);
    expect(normalized.trainValidationGapPercent).toBeCloseTo(2.66106916668108);
    expect(normalized.regularizationEnabled).toBe(true);
    expect(normalized.regularizationType).toBe("l2");
    expect(normalized.regularizationLambda).toBe(0.01);
    expect(normalized.regularizationLambdaKey).toBe("0.01");
    expect(normalized.regularizationLabel).toBe("L2");
  });

  it("survives a summary without config or metrics", () => {
    const normalized = normalizeSummary(
      {},
      {
        experimentName: "old_experiment",
        modifiedAt: 123,
      },
    );

    expect(normalized.experimentName).toBe("old_experiment");
    expect(normalized.modelName).toBe("unknown_model");
    expect(normalized.neuronsProfile).toEqual([]);
    expect(normalized.neuronsProfileLabel).toBe("Unknown");
    expect(normalized.normalizePixels).toBeNull();
    expect(normalized.finalValidationAccuracy).toBeNull();
    expect(normalized.validationErrorPercent).toBeNull();
    expect(normalized.modifiedAt).toBe(123);
    expect(normalized.regularizationEnabled).toBeNull();
    expect(normalized.regularizationType).toBe("unknown");
    expect(normalized.regularizationLambda).toBeNull();
    expect(normalized.regularizationLambdaKey).toBe("unknown");
    expect(normalized.regularizationLabel).toBe("Unknown");
  });

  it("infers the output-only profile for old softmax summaries", () => {
    const summary: RawExperimentSummary = {
      config: {
        model: {
          name: "single_layer_softmax_classifier",
          output_size: 10,
        },
      },
    };

    const normalized = normalizeSummary(summary);

    expect(normalized.neuronsProfile).toEqual([10]);
    expect(normalized.neuronsProfileLabel).toBe("[10]");
  });

  it("normalizes disabled regularization from metadata", () => {
    const summary: RawExperimentSummary = {
      metadata: {
        regularization: {
          enabled: false,
          type: "none",
          lambda: 0.0,
        },
      },
    };

    const normalized = normalizeSummary(summary);

    expect(normalized.regularizationEnabled).toBe(false);
    expect(normalized.regularizationType).toBe("none");
    expect(normalized.regularizationLambda).toBe(0);
    expect(normalized.regularizationLambdaKey).toBe("0");
    expect(normalized.regularizationLabel).toBe("No regularization");
  });
});