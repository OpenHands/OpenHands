import { useTranslation } from "react-i18next";
import { Card } from "#/ui/card";
import { CardTitle } from "#/ui/card-title";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import { BrandButton } from "../settings/brand-button";
import { I18nKey } from "#/i18n/declaration";

export function ContextMenuCTA() {
  const { t } = useTranslation();

  return (
    <Card
      className={cn(
        "w-[342px] rounded-[12px] flex flex-col p-5 bg-gradient-to-b from-black to-[#2d1b4e]",
      )}
    >
      <CardTitle>{t(I18nKey.CTA$ENTERPRISE_TITLE)}</CardTitle>

      <Typography.Text className="text-[#BABBBE]">
        {t(I18nKey.CTA$ENTERPRISE_DESCRIPTION)}
      </Typography.Text>

      <div className="flex gap-2 justify-end mt-auto">
        <BrandButton
          variant="secondary"
          type="button"
          className="w-[147px] border-white text-white hover:bg-white/10"
        >
          {t(I18nKey.CTA$LEARN_MORE)}
        </BrandButton>
        <BrandButton
          variant="primary"
          type="button"
          className="w-[147px] text-black bg-white hover:bg-white/90"
        >
          {t(I18nKey.CTA$SIGN_UP)}
        </BrandButton>
      </div>
    </Card>
  );
}
