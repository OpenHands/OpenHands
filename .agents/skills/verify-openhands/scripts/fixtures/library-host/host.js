import React from "react";
import { createRoot } from "react-dom/client";
import { AgentServerUIRoot } from "@openhands/agent-canvas";
function App() {
  const [count, setCount] = React.useState(0);
  return React.createElement(AgentServerUIRoot, {}, React.createElement("button", { id: "inside", onClick: () => setCount((value) => value + 1) }, "Inside Canvas " + count));
}
createRoot(document.getElementById("root")).render(React.createElement(App));
