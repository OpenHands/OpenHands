---
name: pdflatex
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- pdflatex
---

PdfLatex is a tool that converts Latex sources into PDF. This is specifically very important for researchers, as they use it to publish their findings. It could be installed very easily using Linux terminal, though this seems an annoying task on Windows. Installation commands are given below.

* Install the TexLive base

```
apt-get install texlive-latex-base
```

* Also install the recommended and extra fonts to avoid running into errors, when trying to use pdflatex on latex files with more fonts.

```
apt-get install texlive-fonts-recommended
apt-get install texlive-fonts-extra
```

* Install the extra packages,

```
apt-get install texlive-latex-extra
```

Once installed as above, you may be able to create PDF files from latex sources using PdfLatex as below.
```
pdflatex latex_source_name.tex
```

Ref: http://kkpradeeban.blogspot.com/2014/04/installing-latexpdflatex-on-ubuntu.html
