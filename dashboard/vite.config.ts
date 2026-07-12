import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { defineConfig, type Plugin } from "vite";

const resultsDirectory = path.resolve(__dirname, "../results");

type ExperimentIndexItem = {
  experimentName: string;
  modifiedAt: number;
};

function sendJson(
  response: import("node:http").ServerResponse,
  statusCode: number,
  data: unknown,
) {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(data, null, 2));
}

function getExperimentIndex(): ExperimentIndexItem[] {
  if (!fs.existsSync(resultsDirectory)) {
    return [];
  }

  return fs
    .readdirSync(resultsDirectory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const summaryPath = path.join(resultsDirectory, entry.name, "summary.json");

      if (!fs.existsSync(summaryPath)) {
        return null;
      }

      const summaryStats = fs.statSync(summaryPath);

      return {
        experimentName: entry.name,
        modifiedAt: summaryStats.mtimeMs,
      };
    })
    .filter((item): item is ExperimentIndexItem => item !== null)
    .sort((left, right) =>
      left.experimentName.localeCompare(right.experimentName),
    );
}

function experimentsApiPlugin(): Plugin {
  return {
    name: "experiments-api",

    configureServer(server) {
      server.middlewares.use("/api/experiments", (request, response) => {
        if (request.method !== "GET") {
          sendJson(response, 405, { error: "Method not allowed" });
          return;
        }

        const requestUrl = new URL(request.url ?? "", "http://localhost");
        const experimentNameFromPath = decodeURIComponent(
          requestUrl.pathname.replace(/^\/+/, ""),
        );

        const experimentIndex = getExperimentIndex();
        const allowedExperimentNames = new Set(
          experimentIndex.map((experiment) => experiment.experimentName),
        );

        if (experimentNameFromPath === "") {
          sendJson(response, 200, experimentIndex);
          return;
        }

        const sanitizedExperimentName = path.basename(experimentNameFromPath);

        if (
          sanitizedExperimentName !== experimentNameFromPath ||
          !allowedExperimentNames.has(sanitizedExperimentName)
        ) {
          sendJson(response, 404, { error: "Experiment not found" });
          return;
        }

        const summaryPath = path.join(
          resultsDirectory,
          sanitizedExperimentName,
          "summary.json",
        );

        try {
          const summaryContent = fs.readFileSync(summaryPath, "utf-8");

          response.statusCode = 200;
          response.setHeader("Content-Type", "application/json");
          response.end(summaryContent);
        } catch {
          sendJson(response, 500, { error: "Failed to read summary.json" });
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), experimentsApiPlugin()],
});
