#!/bin/bash

SRC_DIR="src"
OUT_DIR="out"
AUX_DIR="aux"
MAIN_TEX="00_athena.tex"
BASE_NAME="${MAIN_TEX%.tex}"

mkdir -p "$OUT_DIR"
mkdir -p "$AUX_DIR"

# Tell LaTeX to search for inputs in the src/ directory
export TEXINPUTS=.:$SRC_DIR:

pdflatex -output-directory="$AUX_DIR" "$SRC_DIR/$MAIN_TEX"

# Copy .bib to aux to simplify biber
cp "$SRC_DIR/"*.bib "$AUX_DIR/" 2>/dev/null

# Run biber in AUX_DIR
(cd "$AUX_DIR" && biber "$BASE_NAME")

pdflatex -output-directory="$AUX_DIR" "$SRC_DIR/$MAIN_TEX"
pdflatex -output-directory="$AUX_DIR" "$SRC_DIR/$MAIN_TEX"

# Move PDF
mv "$AUX_DIR/$BASE_NAME.pdf" "$OUT_DIR/"