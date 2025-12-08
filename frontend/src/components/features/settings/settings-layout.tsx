import { useState } from "react";
import { MobileHeader } from "./mobile-header";
import { SettingsNavigation } from "./settings-navigation";
import { SettingsNavItem } from "#/constants/settings-nav";

interface SettingsLayoutProps {
  children: React.ReactNode;
  navigationItems: SettingsNavItem[];
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
      {/* Mobile header */}
      <MobileHeader
        isMobileMenuOpen={isMobileMenuOpen}
        onToggleMenu={toggleMobileMenu}
      />
      {/* Desktop layout with navigation and main content */}
      <div className="flex flex-1 overflow-hidden gap-10">
        {/* Navigation */}
        <SettingsNavigation
          isMobileMenuOpen={isMobileMenuOpen}
          onCloseMobileMenu={closeMobileMenu}
          navigationItems={navigationItems}
        />
        {/* Main content */}
        <main className="flex-1 overflow-auto custom-scrollbar-always">
          {children}
        </main>
      </div>
    </div>
  );
}
