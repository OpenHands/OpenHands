import { finishTestRun } from "./done-marker-reporter";

// onEnd runs after webServer teardown; CI needs this signal before it can hang.
export default function doneMarkerTeardown() {
  finishTestRun();
}
