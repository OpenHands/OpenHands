# Centralized Modal Management

## Overview

All modals in OpenHands are managed through a centralized Zustand store and rendered via a single `ModalRoot` component using React portals. This eliminates duplicated backdrop/ESC handling logic and ensures consistent modal behavior across the application.

## Relevant Files

- `src/stores/modal-store/modal-store.ts` - Zustand store managing modal stack
- `src/stores/modal-store/types.ts` - Modal type definitions and props
- `src/components/shared/modals/modal-orchestrator.tsx` - Single portal renderer with backdrop/ESC handling
- Individual modal components (e.g., `confirm-delete-modal.tsx`, `feedback-modal.tsx`)

## Basic Usage

### Opening a Modal

```typescript
import { useModalStore } from "#/stores/modal-store";

function MyComponent() {
  const openModal = useModalStore((state) => state.openModal);

  return (
    <button onClick={() => openModal("confirm-delete", {
      conversationTitle: "My Chat",
      onConfirm: () => handleDelete(),
    })}>
      Delete
    </button>
  );
}
```

> [!IMPORTANT]
> **Use selector form** - `useModalStore((state) => state.openModal)` prevents unnecessary re-renders. Avoid `const { openModal } = useModalStore()`.

### Closing a Modal

Modals auto-close on ESC key or backdrop click. For explicit close buttons:

```typescript
const closeModal = useModalStore((state) => state.closeModal);

<button onClick={closeModal}>Cancel</button>
```

## Adding a New Modal

### 1. Define Type and Props

In `src/stores/modal-store/types.ts`:

```typescript
interface ModalCoreProps {
  // ... existing types

  "my-new-modal": {
    title: string;
    onConfirm: () => void;
  };
}
```

### 2. Create Modal Component

```typescript
// src/components/features/my-feature/my-new-modal.tsx
export function MyNewModal({
  title,
  onConfirm,
  onClose,
}: {
  title: string;
  onConfirm: () => void;
  onClose: () => void;
}) {
  return (
    <ModalBody>
      <h2>{title}</h2>
      <button onClick={() => { onConfirm(); onClose(); }}>Confirm</button>
      <button onClick={onClose}>Cancel</button>
    </ModalBody>
  );
}
```

> [!NOTE]
> Modal components only render content. `ModalRoot` handles visibility, backdrop, and ESC key.

### 3. Register in ModalRoot

In `src/components/shared/modals/modal-orchestrator.tsx`:

```typescript
import { MyNewModal } from "#/components/features/my-feature/my-new-modal";

// Add to renderModal switch:
case "my-new-modal": {
  const props = modal.props as ModalConfigMap["my-new-modal"];
  return (
    <MyNewModal
      title={props.title}
      onConfirm={() => { props.onConfirm(); onClose(); }}
      onClose={onClose}
    />
  );
}
```

### 4. Use It

```typescript
openModal("my-new-modal", {
  title: "Confirm Action",
  onConfirm: () => console.log("Confirmed"),
});
```

## Common Patterns

### Modal Stacking

Multiple modals can be open simultaneously with automatic z-index handling:

```typescript
openModal("confirm-delete", { /* ... */ });
openModal("confirmation", { message: "Save changes?" }); // Renders on top
// ESC closes topmost modal first
```

### Updating Modal Props

Use `replaceModal` to update props of an already-open modal (useful for loading states):

```typescript
const replaceModal = useModalStore((state) => state.replaceModal);

const handleSubmit = async () => {
  replaceModal("my-modal", { isLoading: true });
  try {
    await submitData();
    closeModal();
  } catch (error) {
    replaceModal("my-modal", { isLoading: false, error: error.message });
  }
};
```

### Disabling ESC/Backdrop Close

```typescript
openModal("critical-modal", {
  // ...props
  closeOnEscape: false,
  closeOnBackdrop: false,
});
```

## Anti-patterns to Avoid

### Managing visibility inside modals

```typescript
// ❌ Don't do this
export function MyModal() {
  const [isOpen, setIsOpen] = useState(false);
  if (!isOpen) return null;
  return <ModalBody>...</ModalBody>;
}

// ✅ Do this - ModalRoot handles visibility
export function MyModal({ onClose }) {
  return <ModalBody>...</ModalBody>;
}
```

### Custom backdrop wrappers

```typescript
// ❌ Don't wrap in backdrop
export function MyModal() {
  return (
    <ModalBackdrop>
      <ModalBody>...</ModalBody>
    </ModalBackdrop>
  );
}

// ✅ ModalRoot provides backdrop
export function MyModal({ onClose }) {
  return <ModalBody>...</ModalBody>;
}
```

## Currently Supported Modals

| Modal | Description |
|-------|-------------|
| `confirm-delete` | Delete conversation confirmation |
| `confirm-stop` | Stop conversation confirmation |
| `exit-conversation` | Exit conversation warning |
| `feedback` | Thumbs up/down feedback form |

## Best Practices

- **Use selector form** - `useModalStore((state) => state.openModal)` for specific values
- **Keep modals simple** - Focus on UI, not business logic
- **Pass callbacks** - Let parent components handle actions
- **Trust the types** - TypeScript enforces correct props per modal
- **Test in isolation** - Unit test modal components without the store

## See Also

- [Testing with React Router](__tests__/router.md) - Testing patterns
- [MSW Guide](__tests__/MSW.md) - API mocking in tests
