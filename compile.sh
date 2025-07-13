#!/bin/bash

SRC_DIR="src"
OUT_DIR="out"
MAIN_TEX="00_athena.tex"
BASE_NAME="${MAIN_TEX%.tex}"

mkdir -p "$OUT_DIR"

# Go to src directory where all inputs are
cd "$SRC_DIR" || exit 1

# Run pdflatex (creates aux, bcf, etc)
pdflatex "$MAIN_TEX"

# Run biber (processes citations)
biber "$BASE_NAME"

# Run pdflatex twice more to resolve citations and references
pdflatex "$MAIN_TEX"
pdflatex "$MAIN_TEX"

# Move generated PDF to OUT_DIR (relative to project root)
mv "${BASE_NAME}.pdf" "../${OUT_DIR}/"

echo "Compilation finished, PDF is at ${OUT_DIR}/${BASE_NAME}.pdf"