# @openhands/ui

A modern React component library built with TypeScript and Tailwind CSS.

## Installation

Choose your preferred package manager:

```bash
# npm
npm install @openhands/ui

# yarn
yarn add @openhands/ui

# pnpm
pnpm add @openhands/ui

# bun
bun add @openhands/ui
```

## Quick Start

```tsx
import { Button, Typography } from "@openhands/ui";
import "@openhands/ui/styles";

function App() {
  return (
    <div>
      <Typography.H1>Hello World</Typography.H1>
      <Button variant="primary">Get Started</Button>
    </div>
  );
}
```

## Components

| Component         | Description                                              |
| ----------------- | -------------------------------------------------------- |
| `Accordion`       | Collapsible content sections with single or multi-expand |
| `Button`          | Interactive button with multiple variants                |
| `Checkbox`        | Checkbox input with label support                        |
| `Chip`            | Display tags or labels                                   |
| `Dialog`          | Modal overlay with focus trap and dismiss support        |
| `Divider`         | Visual separator                                         |
| `Icon`            | Icon wrapper component                                   |
| `Input`           | Text input field                                         |
| `InteractiveChip` | Clickable chip component                                 |
| `RadioGroup`      | Radio button group                                       |
| `RadioOption`     | Individual radio option                                  |
| `Scrollable`      | Scrollable container                                     |
| `Select`          | Dropdown select input with search and custom options     |
| `Spinner`         | Loading indicator with determinate and indeterminate modes |
| `Tabs`            | Tabbed navigation with overflow scrolling                |
| `Toast`           | Temporary notification messages                          |
| `Toggle`          | Toggle switch                                            |
| `Tooltip`         | Tooltip overlay                                          |
| `Typography`      | Text components (H1-H6, Text, Code)                     |

## Development

Use your preferred package manager to install dependencies and run the development server. We recommend using [Bun](https://bun.sh) for a fast development experience.

**Note**: If you plan to make dependency changes and submit a PR, you must use Bun during development.

```bash
# Install dependencies
bun install

# Start Storybook
bun run dev

# Build package
bun run build
```

### Testing Locally Without Publishing

To test the package in another project without publishing to npm:

```bash
# Build the package:
bun run build

# Create a local package:
# This generates a `.tgz` file in the current directory.
bun pm pack

# Install in your target project:
# Replace `path/to/openhands-ui-x.x.x.tgz` with the actual path to the generated `.tgz` file.
npm install path/to/openhands-ui-x.x.x.tgz
```

## Publishing

This package is automatically published to npm **when a version bump is merged to the main branch**. See [PUBLISHING.md](./PUBLISHING.md) for detailed information about the publishing process.

## License

MIT
