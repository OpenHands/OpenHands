import { BackendFormModal } from "./backend-form-modal";

interface AddBackendModalProps {
  onClose: () => void;
}

export function AddBackendModal({ onClose }: AddBackendModalProps) {
  return <BackendFormModal mode="add" onClose={onClose} />;
}
