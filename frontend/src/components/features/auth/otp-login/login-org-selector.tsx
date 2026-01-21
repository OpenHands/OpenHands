import { cn } from "@heroui/react";
import { Dispatch, SetStateAction, useState } from "react";
import { useTranslation } from "react-i18next";

interface LoginOrgSelectorProps {
  setIsOrgSelected: Dispatch<SetStateAction<boolean>>;
}

function LoginOrgSelector({ setIsOrgSelected }: LoginOrgSelectorProps) {
  const { t } = useTranslation();

  const orgs = ["org1", "org2"]; // TODO:: replace with API-driven orgs

  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);
  const [rememberLastOrg, setRememberLastOrg] = useState(false);

  const handleLogin = () => {
    if (!selectedOrgId) return;

    // TODO:: store/persist org settings
    // const payload = {
    //   orgId: selectedOrgId,
    //   rememberLastOrg,
    // };

    setIsOrgSelected(true);
  };

  return (
    <div className="flex flex-col w-[700px] text-left gap-[32px]">
      <span className="text-[39px] font-medium">{t("Login as")}</span>

      {/* Org list */}
      <div className="flex flex-col gap-[10px]">
        {orgs.map((org) => {
          const isSelected = selectedOrgId === org;

          return (
            <button
              key={org}
              type="button"
              onClick={() => setSelectedOrgId(org)}
              className={cn(
                "pt-[32px] pb-[32px] px-[20px] w-full rounded-[12px] border-[2px] text-left transition-colors",
                "focus:outline-none focus:ring-2 focus:ring-white",
                isSelected ? "border-white" : "border-[#727987]",
              )}
            >
              {org}
            </button>
          );
        })}

        {/* Remember org */}
        <label className="flex items-center gap-[8px] cursor-pointer">
          <input
            type="checkbox"
            checked={rememberLastOrg}
            onChange={(e) => setRememberLastOrg(e.target.checked)}
            className="h-[16px] w-[16px] rounded-[4px] border-[#D4D4D4] bg-[#FFFFFF] accent-white"
          />
          <span className="text-[12px] font-[400] text-[#A3A3A3]">
            {t("Always login to last used")}
          </span>
        </label>
      </div>

      {/* Submit */}
      <button
        type="button"
        onClick={handleLogin}
        disabled={!selectedOrgId}
        className={cn(
          "h-[40px] w-[147px] rounded-[4px] px-[10px] text-sm font-medium",
          "bg-white text-black",
          "disabled:opacity-50 disabled:cursor-not-allowed",
        )}
      >
        {t("Login")}
      </button>
    </div>
  );
}

export default LoginOrgSelector;
