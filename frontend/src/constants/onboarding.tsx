import { I18nKey } from "#/i18n/declaration";

export type OnboardingAppMode = "cloud" | "self-hosted" | "oss";

interface BaseOnboardingQuestion {
  id: string;
  app_mode: OnboardingAppMode[];
  questionKey: I18nKey;
  subtitleKey?: I18nKey;
}

interface InputQuestion extends BaseOnboardingQuestion {
  type: "input";
  inputOptions: { key: I18nKey; id: string }[];
}

interface SingleSelectQuestion extends BaseOnboardingQuestion {
  type: "single";
  answerOptions: { key: I18nKey; id: string }[];
}

interface MultiSelectQuestion extends BaseOnboardingQuestion {
  type: "multi";
  answerOptions: { key: I18nKey; id: string }[];
}

export type OnboardingQuestion =
  | InputQuestion
  | SingleSelectQuestion
  | MultiSelectQuestion;

export const ONBOARDING_FORM: OnboardingQuestion[] = [
  {
    id: "org_name",
    type: "input",
    app_mode: ["self-hosted"],
    questionKey: I18nKey.ONBOARDING$ORG_NAME_QUESTION,
    inputOptions: [
      { key: I18nKey.ONBOARDING$ORG_NAME_INPUT_NAME, id: "org_name" },
      { key: I18nKey.ONBOARDING$ORG_NAME_INPUT_DOMAIN, id: "org_domain" },
    ],
  },
  {
    id: "org_size",
    type: "single",
    app_mode: ["cloud", "self-hosted"],
    questionKey: I18nKey.ONBOARDING$ORG_SIZE_QUESTION,
    subtitleKey: I18nKey.ONBOARDING$SELECT_ONE_SUBTITLE,
    answerOptions: [
      { key: I18nKey.ONBOARDING$ORG_SIZE_SOLO, id: "solo" },
      { key: I18nKey.ONBOARDING$ORG_SIZE_2_10, id: "org_2_10" },
      { key: I18nKey.ONBOARDING$ORG_SIZE_11_50, id: "org_11_50" },
      { key: I18nKey.ONBOARDING$ORG_SIZE_51_200, id: "org_51_200" },
      { key: I18nKey.ONBOARDING$ORG_SIZE_200_PLUS, id: "org_200_plus" },
    ],
  },
  {
    id: "use_case",
    type: "multi",
    app_mode: ["cloud", "self-hosted"],
    questionKey: I18nKey.ONBOARDING$USE_CASE_QUESTION,
    subtitleKey: I18nKey.ONBOARDING$SELECT_MULTIPLE_SUBTITLE,
    answerOptions: [
      { key: I18nKey.ONBOARDING$USE_CASE_NEW_FEATURES, id: "new_features" },
      {
        key: I18nKey.ONBOARDING$USE_CASE_APP_FROM_SCRATCH,
        id: "app_from_scratch",
      },
      { key: I18nKey.ONBOARDING$USE_CASE_FIXING_BUGS, id: "fixing_bugs" },
      { key: I18nKey.ONBOARDING$USE_CASE_REFACTORING, id: "refactoring" },
      {
        key: I18nKey.ONBOARDING$USE_CASE_AUTOMATING_TASKS,
        id: "automating_tasks",
      },
      { key: I18nKey.ONBOARDING$USE_CASE_NOT_SURE, id: "not_sure" },
    ],
  },
  {
    id: "role",
    type: "single",
    app_mode: ["cloud"],
    questionKey: I18nKey.ONBOARDING$ROLE_QUESTION,
    subtitleKey: I18nKey.ONBOARDING$SELECT_ONE_SUBTITLE,
    answerOptions: [
      {
        key: I18nKey.ONBOARDING$ROLE_SOFTWARE_ENGINEER,
        id: "software_engineer",
      },
      {
        key: I18nKey.ONBOARDING$ROLE_ENGINEERING_MANAGER,
        id: "engineering_manager",
      },
      { key: I18nKey.ONBOARDING$ROLE_CTO_FOUNDER, id: "cto_founder" },
      {
        key: I18nKey.ONBOARDING$ROLE_PRODUCT_OPERATIONS,
        id: "product_operations",
      },
      { key: I18nKey.ONBOARDING$ROLE_STUDENT_HOBBYIST, id: "student_hobbyist" },
      { key: I18nKey.ONBOARDING$ROLE_OTHER, id: "other" },
    ],
  },
];
