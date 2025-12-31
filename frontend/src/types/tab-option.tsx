enum TabOption {
  PLANNER = "planner",
  BROWSER = "browser",
  VSCODE = "vscode",
}

type TabType = TabOption.PLANNER | TabOption.BROWSER | TabOption.VSCODE;
const AllTabs = [TabOption.VSCODE, TabOption.BROWSER, TabOption.PLANNER];

export { AllTabs, TabOption, type TabType };
