const fs = require("fs");
const path = require("path");

const translationPath = path.join(__dirname, "../src/i18n/translation.json");
const translations = JSON.parse(fs.readFileSync(translationPath, "utf8"));

const faOverrides = {
  "APP$TITLE": "سامانه",
  "BROWSER$TITLE": "مرورگر",
  "BROWSER$EMPTY_MESSAGE": "مرورگری برای نمایش در دسترس نیست.",
  "SETTINGS$TITLE": "تنظیمات",
  "CONVERSATION$START_NEW": "شروع گفت‌وگوی جدید",
  "CONVERSATION$REPOSITORY": "مخزن",
  "CONVERSATION$BRANCH": "شاخه",
  "CONVERSATION$GIT_PROVIDER": "ارائه‌دهنده گیت",
  "WORKSPACE$TERMINAL_TAB_LABEL": "ترمینال",
  "WORKSPACE$BROWSER_TAB_LABEL": "مرورگر",
  "WORKSPACE$JUPYTER_TAB_LABEL": "ژوپیتر",
  "WORKSPACE$CODE_EDITOR_TAB_LABEL": "ویرایشگر کد",
  "WORKSPACE$TITLE": "فضای کاری",
  "TERMINAL$WAITING_FOR_CLIENT": "در انتظار آماده‌شدن کلاینت...",
  "CODE_EDITOR$FILE_SAVED_SUCCESSFULLY": "فایل با موفقیت ذخیره شد",
  "CODE_EDITOR$SAVING_LABEL": "در حال ذخیره...",
  "CODE_EDITOR$SAVE_LABEL": "ذخیره",
  "CODE_EDITOR$OPTIONS": "گزینه‌ها",
  "CODE_EDITOR$FILE_SAVE_ERROR": "ذخیره فایل ناموفق بود",
  "CODE_EDITOR$EMPTY_MESSAGE": "فایلی برای نمایش انتخاب نشده است",
  "HOME$LAUNCH_FROM_SCRATCH": "شروع از صفر",
  "HOME$READ_THIS": "این را بخوانید",
  "HOME$CONNECT_PROVIDER_MESSAGE": "برای شروع، ارائه‌دهنده مدل را متصل کنید.",
  "HOME$LETS_START_BUILDING": "بیایید شروع کنیم",
  "HOME$OPENHANDS_DESCRIPTION":
    "OpenHands ساخت و نگه‌داری نرم‌افزار را با توسعه مبتنی بر هوش مصنوعی ساده می‌کند.",
  "HOME$NOT_SURE_HOW_TO_START": "نمی‌دانید از کجا شروع کنید؟",
  "HOME$CONNECT_TO_REPOSITORY": "اتصال به مخزن",
  "HOME$CONNECT_TO_REPOSITORY_TOOLTIP":
    "اگر می‌خواهید روی یک مخزن عمومی کار کنید، می‌توانید نشانی عمومی GitHub را وارد کنید.",
  "HOME$LOADING": "در حال بارگذاری",
  "HOME$LOADING_REPOSITORIES": "در حال بارگذاری مخزن‌ها",
  "HOME$SEARCHING_REPOSITORIES": "در حال جست‌وجوی مخزن‌ها",
  "HOME$LOADING_MORE_REPOSITORIES": "در حال بارگذاری موارد بیشتر",
  "HOME$FAILED_TO_LOAD_REPOSITORIES": "بارگذاری مخزن‌ها ناموفق بود",
  "HOME$LOADING_BRANCHES": "در حال بارگذاری شاخه‌ها",
  "HOME$FAILED_TO_LOAD_BRANCHES": "بارگذاری شاخه‌ها ناموفق بود",
  "HOME$OPEN_ISSUE": "باز کردن issue",
  "HOME$FIX_FAILING_CHECKS": "رفع checkهای ناموفق",
  "HOME$RESOLVE_MERGE_CONFLICTS": "حل تعارض‌های ادغام",
  "HOME$RESOLVE_UNRESOLVED_COMMENTS": "رسیدگی به نظرهای حل‌نشده",
  "HOME$LAUNCH": "اجرا",
  "SETTINGS$ADVANCED": "پیشرفته",
  "SETTINGS$BASE_URL": "نشانی پایه",
  "SETTINGS$AGENT": "عامل",
  "SETTINGS$LANGUAGE": "زبان",
  "SETTINGS$LLM_SETTINGS": "تنظیمات مدل",
  "SETTINGS$GIT_SETTINGS": "تنظیمات گیت",
  "SETTINGS$GIT_SETTINGS_DESCRIPTION":
    "نحوه اتصال OpenHands به مخزن‌ها و ارائه‌دهنده‌های گیت را مدیریت کنید.",
  "SETTINGS$SOUND_NOTIFICATIONS": "اعلان صوتی",
  "SETTINGS$MAX_BUDGET_PER_TASK": "حداکثر بودجه برای هر کار",
  "SETTINGS$MAX_BUDGET_PER_CONVERSATION": "حداکثر بودجه برای هر گفت‌وگو",
  "SETTINGS$SAVING": "در حال ذخیره...",
  "SETTINGS$SAVE_CHANGES": "ذخیره تغییرات",
  "ACTION$PUSH_TO_BRANCH": "ارسال به شاخه",
  "ACTION$PUSH_CREATE_PR": "ارسال و ایجاد PR",
  "ACTION$PUSH_CHANGES_TO_PR": "ارسال تغییرات به PR",
  "ANALYTICS$TITLE": "تحلیل‌گر",
  "ANALYTICS$DESCRIPTION":
    "به ما اجازه دهید داده‌های ناشناس استفاده را برای بهبود OpenHands دریافت کنیم.",
  "ANALYTICS$SEND_ANONYMOUS_DATA": "ارسال داده‌های ناشناس",
  "ANALYTICS$CONFIRM_PREFERENCES": "تأیید ترجیحات",
  "BUTTON$COPY": "کپی در کلیپ‌بورد",
  "BUTTON$COPIED": "در کلیپ‌بورد کپی شد",
  "BUTTON$SAVE": "ذخیره",
  "BUTTON$CLOSE": "بستن",
  "BUTTON$END_SESSION": "پایان نشست",
  "BUTTON$LAUNCH": "اجرا",
  "BUTTON$CANCEL": "انصراف",
  "BUTTON$ADD": "افزودن",
  "BUTTON$DISCONNECT": "قطع اتصال",
  "MODAL$CONFIRM_RESET_TITLE": "مطمئن هستید؟",
  "MODAL$CONFIRM_RESET_MESSAGE":
    "این کار تنظیمات فعلی را بازنشانی می‌کند. ادامه می‌دهید؟",
  "MODAL$END_SESSION_TITLE": "پایان نشست",
  "MODAL$END_SESSION_MESSAGE":
    "پس از پایان این نشست، ادامه‌ی آن در همین فضا ممکن نخواهد بود.",
  "EXIT_PROJECT$CONFIRM": "خروج از پروژه",
  "EXIT_PROJECT$TITLE": "ترک پروژه",
  "LANGUAGE$LABEL": "زبان",
  "LLM$PROVIDER": "ارائه‌دهنده مدل",
  "LLM$SELECT_PROVIDER_PLACEHOLDER": "یک ارائه‌دهنده انتخاب کنید",
  "LLM$MODEL": "مدل LLM",
  "LLM$SELECT_MODEL_PLACEHOLDER": "یک مدل را انتخاب کنید",
  "API$KEY": "کلید API",
  "API$DONT_KNOW_KEY": "کلید API خود را نمی‌دانید؟",
  "GITHUB$TOKEN_LABEL": "توکن GitHub",
  "GITHUB$HOST_LABEL": "میزبان GitHub",
  "GITHUB$TOKEN_OPTIONAL": "اختیاری",
  "GITHUB$GET_TOKEN": "دریافت توکن",
  "GITHUB$TOKEN_HELP_TEXT":
    "اگر روی مخزن‌های خصوصی یا عملیات نوشتن کار می‌کنید، یک توکن وارد کنید.",
  "GITHUB$TOKEN_LINK_TEXT": "ایجاد توکن GitHub",
  "GITHUB$INSTRUCTIONS_LINK_TEXT": "راهنمای تنظیم",
  "GITHUB$TOKEN_INVALID": "توکن GitHub معتبر نیست",
  "GITHUB$CONFIGURE_REPOS": "پیکربندی مخزن‌ها",
  "SLACK$INSTALL_APP": "نصب برنامه Slack",
  "COMMON$STATUS": "وضعیت",
  "COMMON$HERE": "اینجا",
  "COMMON$CLICK_FOR_INSTRUCTIONS": "برای راهنما کلیک کنید",
  "STATUS$CONNECTED_TO_SERVER": "به سرور متصل شد",
  "PROJECT$NEW_PROJECT": "پروژه جدید",
  "AUTH$LOGGING_BACK_IN": "در حال ورود دوباره به OpenHands...",
  "SECURITY$LOW_RISK": "ریسک: پایین",
  "SECURITY$MEDIUM_RISK": "ریسک: متوسط",
  "SECURITY$HIGH_RISK": "ریسک: بالا",
  "SECURITY$UNKNOWN_RISK": "ریسک: نامشخص",
  "BRANDING$OPENHANDS": "Gantor OpenHands",
  "BRANDING$OPENHANDS_LOGO": "نشان Gantor OpenHands",
  "ERROR$GENERIC": "خطایی رخ داد",
  "VSCODE$TITLE": "VS Code",
  "VSCODE$LOADING": "در حال بارگذاری VS Code...",
  "VSCODE$URL_NOT_AVAILABLE": "نشانی VS Code در دسترس نیست",
  "VSCODE$FETCH_ERROR": "دریافت VS Code ناموفق بود",
  "VSCODE$OPEN_IN_NEW_TAB": "باز کردن در تب جدید",
  "CONFIGURATION$MODAL_TITLE": "پیکربندی OpenHands",
  "CONFIGURATION$MODEL_SELECT_LABEL": "مدل",
  "CONFIGURATION$MODEL_SELECT_PLACEHOLDER": "مدل را انتخاب کنید",
  "CONFIGURATION$AGENT_SELECT_LABEL": "عامل",
  "CONFIGURATION$AGENT_SELECT_PLACEHOLDER": "عامل را انتخاب کنید",
  "CONFIGURATION$LANGUAGE_SELECT_LABEL": "زبان",
  "CONFIGURATION$LANGUAGE_SELECT_PLACEHOLDER": "زبان را انتخاب کنید",
  "CONFIGURATION$SECURITY_SELECT_LABEL": "سطح امنیت",
  "CONFIGURATION$SECURITY_SELECT_PLACEHOLDER": "سطح امنیت را انتخاب کنید",
  "CONFIGURATION$MODAL_CLOSE_BUTTON_LABEL": "بستن",
  "CONFIGURATION$MODAL_SAVE_BUTTON_LABEL": "ذخیره",
  "CONFIGURATION$MODAL_RESET_BUTTON_LABEL": "بازنشانی",
};

for (const [key, value] of Object.entries(translations)) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    if (!("fa" in value) && "en" in value) {
      value.fa = value.en;
    }
  }
}

for (const [key, value] of Object.entries(faOverrides)) {
  if (!translations[key]) {
    continue;
  }
  translations[key].fa = value;
}

fs.writeFileSync(translationPath, JSON.stringify(translations, null, 2) + "\n");
console.log("fa locale seeded");
