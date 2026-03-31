import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { getRandomTip } from "#/utils/tips";
import { Provider } from "#/types/settings";

interface RandomTipProps {
  gitProvider?: Provider | null;
}

export function RandomTip({ gitProvider = null }: RandomTipProps) {
  const { t } = useTranslation();
  const [randomTip, setRandomTip] = React.useState(() =>
    getRandomTip(gitProvider),
  );

  // Update the random tip when the active conversation provider changes.
  React.useEffect(() => {
    setRandomTip(getRandomTip(gitProvider));
  }, [gitProvider]);

  return (
    <div>
      <h4 className="font-bold">{t(I18nKey.TIPS$PROTIP)}:</h4>
      <p>
        {t(randomTip.key)}
        {randomTip.link && (
          <>
            {" "}
            <a
              href={randomTip.link}
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              {t(I18nKey.TIPS$LEARN_MORE)}
            </a>
          </>
        )}
      </p>
    </div>
  );
}
