import React from "react";

export default function CrossAppRedirect() {
  React.useEffect(() => {
    window.location.replace(window.location.href);
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center bg-base">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white" />
    </div>
  );
}
