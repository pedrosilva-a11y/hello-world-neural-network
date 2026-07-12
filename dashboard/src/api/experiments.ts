import { normalizeSummary } from "../lib/normalizeSummary";
import type {
  ExperimentIndexItem,
  LoadedExperiments,
  RawExperimentSummary,
} from "../types/summary";

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Request failed for ${url}: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function fetchExperimentIndex(): Promise<ExperimentIndexItem[]> {
  return fetchJson<ExperimentIndexItem[]>("/api/experiments");
}

export async function fetchExperimentSummary(
  experimentName: string,
): Promise<RawExperimentSummary> {
  return fetchJson<RawExperimentSummary>(
    `/api/experiments/${encodeURIComponent(experimentName)}`,
  );
}

export async function fetchLoadedExperiments(): Promise<LoadedExperiments> {
  const experimentIndex = await fetchExperimentIndex();

  const settledSummaries = await Promise.allSettled(
    experimentIndex.map(async (indexItem) => {
      const summary = await fetchExperimentSummary(indexItem.experimentName);

      return normalizeSummary(summary, indexItem);
    }),
  );

  const experiments = [];
  const failedExperimentNames = [];

  for (let index = 0; index < settledSummaries.length; index += 1) {
    const settledSummary = settledSummaries[index];
    const experimentName = experimentIndex[index].experimentName;

    if (settledSummary.status === "fulfilled") {
      experiments.push(settledSummary.value);
    } else {
      failedExperimentNames.push(experimentName);
    }
  }

  experiments.sort((left, right) =>
    left.experimentName.localeCompare(right.experimentName),
  );

  return {
    experiments,
    failedExperimentNames,
  };
}
