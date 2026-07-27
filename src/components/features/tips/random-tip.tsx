import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { getRandomTip } from "#/utils/tips";
import LightbulbIcon from "#/icons/lightbulb.svg?react";

export function RandomTip() {
  const { t } = useTranslation("openhands");
  const [randomTip, setRandomTip] = React.useState(getRandomTip());

  React.useEffect(() => {
    setRandomTip(getRandomTip());
  }, []);

  return (
    <div className="w-full bg-tertiary p-4 text-left">
      <div className="flex items-start gap-3">
        <LightbulbIcon
          className="mt-0.5 h-4 w-4 shrink-0 fill-[var(--oh-muted)]"
          aria-hidden
        />
        <div className="min-w-0">
          <h4 className="text-sm font-semibold text-[var(--oh-text-tertiary)]">
            {t(I18nKey.TIPS$PROTIP)}
          </h4>
          <p className="text-sm font-normal leading-5 text-[var(--oh-muted)]">
            {t(randomTip.key)}
            {randomTip.link ? (
              <>
                {" "}
                <a
                  href={randomTip.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline transition-colors hover:text-white"
                >
                  {t(I18nKey.TIPS$LEARN_MORE)}
                </a>
              </>
            ) : null}
          </p>
        </div>
      </div>
    </div>
  );
}
