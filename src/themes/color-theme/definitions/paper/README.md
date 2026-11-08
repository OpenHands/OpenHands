# Newsreader

Paper uses the unmodified Newsreader roman variable font by Production Type,
licensed under the SIL Open Font License 1.1. `newsreader-OFL.txt` preserves
the full license and copyright. Vite emits it as `assets/newsreader-OFL.txt`
in both standalone and library builds, alongside the distributed font data.

Source: [Production Type / Newsreader](https://github.com/productiontype/Newsreader/blob/cfcb4f7af0e52c25e8df2a2431814c8e5fe2e155/fonts/variable/woff2/Newsreader%5Bopsz%2Cwght%5D.woff2).
The local filename is `newsreader-roman.woff2`; the full character coverage,
weight axis (200–800), and optical-size axis are retained.

Only display headings use this face. System sans controls and message text,
and monospace code/terminal output, keep their existing fonts. The app serves
the font locally on demand; the library bundles it into its stylesheet.
