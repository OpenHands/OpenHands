import { useState } from "react";
import { MobileHeader } from "./mobile-header";
import { SettingsNavigation } from "./settings-navigation";
import { SettingsNavItem } from "#/constants/settings-nav";

interface SettingsLayoutProps {
  children: React.ReactNode;
  navigationItems: SettingsNavItem[]; // <-- Add this back
}

export function SettingsLayout({
  children,
  navigationItems,
}: SettingsLayoutProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => setIsMobileMenuOpen(!isMobileMenuOpen);
  const closeMobileMenu = () => setIsMobileMenuOpen(false);

  return (
    <div className="flex flex-col h-full px-[14px] pt-8">
      <MobileHeader
        isMobileMenuOpen={isMobileMenuOpen}
        onToggleMenu={toggleMobileMenu}
      />

      <div className="flex flex-1 overflow-hidden gap-10">
        <SettingsNavigation
          isMobileMenuOpen={isMobileMenuOpen}
          onCloseMobileMenu={closeMobileMenu}
          navigationItems={navigationItems} // <-- Send items here
        />

        <main className="flex-1 overflow-auto custom-scrollbar-always">
          {children}
        </main>
      </div>
    </div>
  );
}
