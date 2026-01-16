# Centralized Modal Management

## Overview

All modals in OpenHands are managed through a centralized Zustand store and rendered via a single `ModalRoot` component using React portals. This eliminates duplicated backdrop/ESC handling logic and ensures consistent modal behavior across the application.

## Relevant Files

- `src/stores/modal-store/modal-store.ts` - Zustand store managing modal stack
- `src/stores/modal-store/types.ts` - Modal type definitions and props
- `src/components/shared/modals/modal-root.tsx` - Single portal renderer with backdrop/ESC handling
- Individual modal components (e.g., `settings-modal.tsx`, `feedback-modal.tsx`)

## Basic Usage

### Opening a Modal

```typescript
import { useModalStore } from "#/stores/modal-store/modal-store";

function MyComponent() {
  const openModal = useModalStore((state) => state.openModal);

  return (
    <button onClick={() => openModal("settings", {})}>
      Open Settings
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

<button onClick={() => closeModal("settings")}>Close</button>
```

## Adding a New Modal

### 1. Define Type and Props

In `src/stores/modal-store/types.ts`:

```typescript
export type ModalType =
  | "settings"
  | "feedback"
  | "my-new-modal"; // Add your type

export interface MyNewModalProps {
  title: string;
  onConfirm: () => void;
}

export interface ModalPropsMap {
  settings: SettingsModalProps;
  feedback: FeedbackProps;
  "my-new-modal": MyNewModalProps; // Add mapping
}
```

### 2. Create Modal Component

```typescript
// src/components/shared/modals/my-new-modal.tsx
import { useModalStore } from "#/stores/modal-store/modal-store";

export function MyNewModal({ title, onConfirm }: MyNewModalProps) {
  const closeModal = useModalStore((state) => state.closeModal);

  return (
    <div className="modal-content">
      <h2>{title}</h2>
      <button onClick={() => {
        onConfirm();
        closeModal("my-new-modal");
      }}>
        Confirm
      </button>
      <button onClick={() => closeModal("my-new-modal")}>Cancel</button>
    </div>
  );
}
```

> [!NOTE]
> Modal components only render content. `ModalRoot` handles visibility, backdrop, and ESC key.

### 3. Register in ModalRoot

In `src/components/shared/modals/modal-root.tsx`:

```typescript
import { MyNewModal } from "./my-new-modal";

// Add to render logic:
{modal.type === "my-new-modal" && <MyNewModal {...modal.props} />}
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
openModal("settings", {});
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
    closeModal("my-modal");
  } catch (error) {
    replaceModal("my-modal", { isLoading: false, error: error.message });
  }
};
```

### Conditional Opening

```typescript
// ✅ Preferred - direct user action
<button onClick={() => openModal("settings", {})}>Settings</button>

// ⚠️ Use sparingly - avoid infinite loops
useEffect(() => {
  if (needsVerification && !modalOpen) {
    openModal("email-verification", { userId });
  }
}, [needsVerification]);
```

## Anti-patterns to Avoid

### Managing visibility inside modals

```typescript
// ❌ Don't do this
export function MyModal() {
  const [isOpen, setIsOpen] = useState(false);

  if (!isOpen) return null;

  return <div className="modal-content">...</div>;
}

// ✅ Do this - ModalRoot handles visibility
export function MyModal() {
  return <div className="modal-content">...</div>;
}
```

### Custom backdrop wrappers

```typescript
// ❌ Don't wrap in backdrop
export function MyModal() {
  return (
    <div className="backdrop" onClick={closeModal}>
      <div className="modal-content">...</div>
    </div>
  );
}

// ✅ ModalRoot provides backdrop
export function MyModal() {
  return <div className="modal-content">...</div>;
}
```

### Heavy logic in modal components

```typescript
// ❌ Avoid data fetching in modals
export function MyModal({ userId }: { userId: string }) {
  const [user, setUser] = useState();
  useEffect(() => { fetchUser(userId).then(setUser); }, [userId]);
  return <div>{user?.name}</div>;
}

// ✅ Pass computed data as props
export function MyModal({ user }: { user: User }) {
  return <div>{user.name}</div>;
}
```

## Migrating Existing Modals

### Before (Old Pattern)
```typescript
function OldComponent() {
  const [modalOpen, setModalOpen] = useState(false);

  return (
    <>
      <button onClick={() => setModalOpen(true)}>Open</button>
      {modalOpen && (
        <ModalBackdrop onClose={() => setModalOpen(false)}>
          <div>Modal content</div>
        </ModalBackdrop>
      )}
    </>
  );
}
```

### After (Centralized Pattern)
```typescript
// 1. Define type in types.ts
// 2. Create modal component:
function MyModal() {
  const closeModal = useModalStore((state) => state.closeModal);
  return <div>Modal content</div>;
}

// 3. Register in ModalRoot
// 4. Use in component:
function NewComponent() {
  const openModal = useModalStore((state) => state.openModal);
  return <button onClick={() => openModal("my-modal", {})}>Open</button>;
}
```

**Key Changes**:
- Remove `useState` for modal visibility
- Remove `ModalBackdrop` wrapper
- Call `openModal` instead of `setModalOpen(true)`

## Currently Supported Modals

23 modals across conversation management, settings, API keys, microagents, integrations, authentication, and billing.

See `src/stores/modal-store/types.ts` for the complete list.

## Best Practices

- **Use selector form** - `useModalStore((state) => state.openModal)` for specific values
- **Keep modals simple** - Focus on UI, not business logic
- **Pass computed data** - Don't fetch data inside modals
- **Trust the types** - TypeScript enforces correct props per modal
- **Test in isolation** - Unit test modal components without the store
