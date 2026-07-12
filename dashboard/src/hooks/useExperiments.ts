import { useCallback, useEffect, useState } from "react";

import { fetchLoadedExperiments } from "../api/experiments";
import type { LoadedExperiments } from "../types/summary";

type UseExperimentsState = {
  data: LoadedExperiments;
  isLoading: boolean;
  errorMessage: string | null;
  refresh: () => Promise<void>;
};

const emptyLoadedExperiments: LoadedExperiments = {
  experiments: [],
  failedExperimentNames: [],
};

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return "Unknown error";
}

export function useExperiments(): UseExperimentsState {
  const [data, setData] = useState<LoadedExperiments>(emptyLoadedExperiments);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage(null);

    try {
      const loadedExperiments = await fetchLoadedExperiments();
      setData(loadedExperiments);
    } catch (error) {
      setErrorMessage(getErrorMessage(error));
      setData(emptyLoadedExperiments);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let isActive = true;

    async function loadInitialExperiments() {
      try {
        const loadedExperiments = await fetchLoadedExperiments();

        if (!isActive) {
          return;
        }

        setData(loadedExperiments);
      } catch (error) {
        if (!isActive) {
          return;
        }

        setErrorMessage(getErrorMessage(error));
        setData(emptyLoadedExperiments);
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadInitialExperiments();

    return () => {
      isActive = false;
    };
  }, []);

  return {
    data,
    isLoading,
    errorMessage,
    refresh,
  };
}
