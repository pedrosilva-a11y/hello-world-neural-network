import { useMemo, useState } from "react";

import { getConfigDiffRows } from "../../lib/configDiff";
import type { NormalizedExperiment } from "../../types/summary";

type ConfigDiffProps = {
  experiments: NormalizedExperiment[];
};

function getInitialRightExperimentName(
  experiments: NormalizedExperiment[],
): string {
  return experiments[1]?.experimentName ?? experiments[0]?.experimentName ?? "";
}

export function ConfigDiff({ experiments }: ConfigDiffProps) {
  const [leftExperimentName, setLeftExperimentName] = useState(
    experiments[0]?.experimentName ?? "",
  );
  const [rightExperimentName, setRightExperimentName] = useState(
    getInitialRightExperimentName(experiments),
  );
  const [showUnchanged, setShowUnchanged] = useState(false);

  const leftExperiment =
    experiments.find((experiment) => experiment.experimentName === leftExperimentName) ??
    experiments[0] ??
    null;

  const rightExperiment =
    experiments.find((experiment) => experiment.experimentName === rightExperimentName) ??
    experiments[1] ??
    experiments[0] ??
    null;

  const diffRows = useMemo(() => {
    if (leftExperiment === null || rightExperiment === null) {
      return [];
    }

    const rows = getConfigDiffRows(
      leftExperiment.raw.config,
      rightExperiment.raw.config,
    );

    if (showUnchanged) {
      return rows;
    }

    return rows.filter((row) => row.isDifferent);
  }, [leftExperiment, rightExperiment, showUnchanged]);

  if (experiments.length < 2 || leftExperiment === null || rightExperiment === null) {
    return (
      <section className="dashboard-section">
        <div className="section-header">
          <div>
            <h2>Config diff</h2>
            <p className="muted">
              At least two filtered experiments are needed to compare configs.
            </p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="dashboard-section">
      <div className="section-header section-header-with-controls">
        <div>
          <h2>Config diff</h2>
          <p className="muted">
            Compare two filtered experiments and show changed config keys.
          </p>
        </div>

        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={showUnchanged}
            onChange={(event) => setShowUnchanged(event.target.checked)}
          />
          Show unchanged
        </label>
      </div>

      <div className="diff-controls">
        <label>
          Left experiment
          <select
            value={leftExperiment.experimentName}
            onChange={(event) => setLeftExperimentName(event.target.value)}
          >
            {experiments.map((experiment) => (
              <option
                key={experiment.experimentName}
                value={experiment.experimentName}
              >
                {experiment.experimentName}
              </option>
            ))}
          </select>
        </label>

        <label>
          Right experiment
          <select
            value={rightExperiment.experimentName}
            onChange={(event) => setRightExperimentName(event.target.value)}
          >
            {experiments.map((experiment) => (
              <option
                key={experiment.experimentName}
                value={experiment.experimentName}
              >
                {experiment.experimentName}
              </option>
            ))}
          </select>
        </label>
      </div>

      {diffRows.length === 0 ? (
        <p className="muted">No config differences found.</p>
      ) : (
        <div className="table-scroll">
          <table className="diff-table">
            <thead>
              <tr>
                <th>Config key</th>
                <th>{leftExperiment.experimentName}</th>
                <th>{rightExperiment.experimentName}</th>
              </tr>
            </thead>

            <tbody>
              {diffRows.map((row) => (
                <tr
                  className={row.isDifferent ? "diff-row-changed" : ""}
                  key={row.path}
                >
                  <td className="strong-cell">{row.path}</td>
                  <td>{row.leftValue}</td>
                  <td>{row.rightValue}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
