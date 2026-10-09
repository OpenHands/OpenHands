# Automation interface availability

### AIA-001: Home hides automation entry points without an admitted interface manifest

When `@openhands/extensions` provides no admitted automation interface manifest,
Home omits the recommended, pinned, and running automation sections and the
Getting started checklist omits **Schedule a task**. Other Home content remains
available. The Automate sidebar entry stays hidden and `/automations` remains a
404 until an interface manifest is admitted.
