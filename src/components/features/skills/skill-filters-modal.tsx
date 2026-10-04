import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  ModalBody,
} from "#/components/shared/modals/modal-body";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { SkillFacetRail } from "./skill-facet-rail";
import type { SkillFacetGroup, SkillFacetGroupId } from "./skill-filter";

interface SkillFiltersModalProps {
  groups: SkillFacetGroup[];
  activeCount: number;
  onToggle: (groupId: SkillFacetGroupId, value: string) => void;
  onClearAll: () => void;
  onClose: () => void;
}

export function SkillFiltersModal({
  groups,
  activeCount,
  onToggle,
  onClearAll,
  onClose,
}: SkillFiltersModalProps) {
  const { t } = useTranslation("openhands");

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.SETTINGS$SKILLS_FILTERS_MODAL_TITLE)}
    >
      <ModalBody
        testID="skill-filters-modal"
        className={cn("items-stretch", MODAL_MAX_WIDTH_VIEWPORT)}
      >
        <h3 className="text-lg font-semibold text-foreground">
          {t(I18nKey.SETTINGS$SKILLS_FILTERS_MODAL_TITLE)}
        </h3>

        <div className="max-h-[60vh] overflow-y-auto custom-scrollbar-always">
          <SkillFacetRail groups={groups} onToggle={onToggle} />
        </div>

        <div className="flex items-center justify-end gap-2">
          {activeCount > 0 ? (
            <BrandButton
              type="button"
              variant="secondary"
              testId="skill-filters-modal-clear"
              onClick={onClearAll}
            >
              {t(I18nKey.SETTINGS$SKILLS_CLEAR_FILTERS)}
            </BrandButton>
          ) : null}
          <BrandButton
            type="button"
            variant="primary"
            testId="skill-filters-modal-done"
            onClick={onClose}
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
