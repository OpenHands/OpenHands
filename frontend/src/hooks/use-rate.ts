import React from "react";

interface UseRateProps {
  threshold: number;
}

const DEFAULT_CONFIG: UseRateProps = { threshold: 1000 };

export const useRate = (config = DEFAULT_CONFIG) => {
  const [items, setItems] = React.useState<number[]>([]);
  const [rate, setRate] = React.useState<number | null>(null);
  const [lastUpdated, setLastUpdated] = React.useState<number | null>(null);
  const [isUnderThreshold, setIsUnderThreshold] = React.useState(true);

  /**
   * Record an entry in order to calculate the rate
   * @param entry Entry to record
   *
   * @example
   * record(new Date().getTime());
   */
  const record = (entry: number) => {
    setItems((prev) => [...prev.slice(-1), entry]);
    setLastUpdated(new Date().getTime());
  };

  /**
   * Update the rate based on the last two entries (if available)
   */
  const updateRate = () => {
    if (items.length > 1) {
      const newRate = items[items.length - 1] - items[items.length - 2];
      setRate(newRate);

      if (newRate <= config.threshold) setIsUnderThreshold(true);
      else setIsUnderThreshold(false);
    }
  };

  React.useEffect(() => {
    updateRate();
  }, [items]);

  React.useEffect(() => {
    // Set up an interval to check if the time since the last update exceeds the threshold
    // If it does, set isUnderThreshold to false, otherwise set it to true
    // This ensures that the component can react to periods of inactivity
    const intervalId = setInterval(() => {
      if (lastUpdated !== null) {
        const timeSinceLastUpdate = new Date().getTime() - lastUpdated;
        setIsUnderThreshold(timeSinceLastUpdate <= config.threshold);
      } else {
        setIsUnderThreshold(false);
      }
    }, config.threshold);

    return () => clearInterval(intervalId);
  }, [lastUpdated, config.threshold]);

  return {
    items,
    rate,
    lastUpdated,
    isUnderThreshold,
    record,
  };
};
