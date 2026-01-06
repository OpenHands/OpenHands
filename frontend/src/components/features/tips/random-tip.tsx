import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { getRandomTip } from "#/utils/tips";

export function RandomTip() {
  const { t } = useTranslation();
  const [randomTip, setRandomTip] = React.useState(getRandomTip());

  // Update the random tip when the component mounts
  React.useEffect(() => {
    setRandomTip(getRandomTip());
  }, []);

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
