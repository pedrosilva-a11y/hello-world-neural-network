import { lazy, Suspense, useMemo, useState } from "react";

import { ConfigDiff } from "./components/diff/ConfigDiff";
import { ExperimentDetail } from "./components/detail/ExperimentDetail";
import { FilterSidebar } from "./components/filters/FilterSidebar";
import { MetricCards } from "./components/overview/MetricCards";
import { RankingTable } from "./components/overview/RankingTable";
import {
  emptyDashboardFilters,
  filterExperiments,
  type DashboardFilters,
} from "./lib/filterExperiments";
import { useExperiments } from "./hooks/useExperiments";

const ComparisonCharts = lazy(() =>
  import("./components/comparisons/ComparisonCharts").then((module) => ({
    default: module.ComparisonCharts,
  })),
);

const TrainingCurves = lazy(() =>
  import("./components/curves/TrainingCurves").then((module) => ({
    default: module.TrainingCurves,
  })),
);

function App() {
  const { data, errorMessage, isLoading, refresh } = useExperiments();
  const [filters, setFilters] = useState<DashboardFilters>(emptyDashboardFilters);
  const [selectedExperimentName, setSelectedExperimentName] = useState("");

  const filteredExperiments = useMemo(
    () => filterExperiments(data.experiments, filters),
    [data.experiments, filters],
  );

  const selectedExperiment = useMemo(() => {
    return (
      filteredExperiments.find(
        (experiment) => experiment.experimentName === selectedExperimentName,
      ) ??
      filteredExperiments[0] ??
      null
    );
  }, [filteredExperiments, selectedExperimentName]);

  return (
    <main className="app-shell">
      <header className="dashboard-header">
        <div className="dashboard-pill">Kaggle Digit Recognizer</div>
        <h1>Experiment Dashboard</h1>
        <p>
          Local React dashboard reading neural-network experiment summaries from
          <code> results/*/summary.json</code>.
        </p>
      </header>

      <section className="toolbar">
        <div>
          <h2>Loaded experiments</h2>
          <p className="muted">
            {data.experiments.length} experiments loaded
            {data.failedExperimentNames.length > 0
              ? ` · ${data.failedExperimentNames.length} failed`
              : ""}
          </p>
        </div>

        <button type="button" onClick={() => void refresh()} disabled={isLoading}>
          {isLoading ? "Refreshing..." : "Refresh"}
        </button>
      </section>

      {errorMessage !== null && (
        <section className="alert alert-error">
          <strong>Failed to load dashboard data.</strong>
          <p>{errorMessage}</p>
        </section>
      )}

      {data.failedExperimentNames.length > 0 && (
        <section className="alert alert-warning">
          <strong>Some summaries failed to load:</strong>
          <ul>
            {data.failedExperimentNames.map((experimentName) => (
              <li key={experimentName}>{experimentName}</li>
            ))}
          </ul>
        </section>
      )}

      {isLoading && <p className="muted">Loading experiments...</p>}

      {!isLoading && data.experiments.length === 0 && errorMessage === null && (
        <p className="muted">
          No experiments found. Run a training experiment first.
        </p>
      )}

      {data.experiments.length > 0 && (
        <div className="dashboard-grid">
          <FilterSidebar
            experiments={data.experiments}
            filteredExperimentCount={filteredExperiments.length}
            filters={filters}
            onFiltersChange={setFilters}
          />

          <div className="dashboard-main">
            <MetricCards experiments={filteredExperiments} />

            <Suspense fallback={<p className="muted">Loading charts...</p>}>
              <ComparisonCharts experiments={filteredExperiments} />
              <TrainingCurves experiments={filteredExperiments} />
            </Suspense>

            <RankingTable experiments={filteredExperiments} />

            <ExperimentDetail
              experiments={filteredExperiments}
              selectedExperiment={selectedExperiment}
              onSelectedExperimentNameChange={setSelectedExperimentName}
            />

            <ConfigDiff experiments={filteredExperiments} />
          </div>
        </div>
      )}
    </main>
  );
}

export default App;