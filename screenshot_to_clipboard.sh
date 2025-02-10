#!/bin/bash

# Create a temporary file to store the screenshot
TMP_IMG=$(mktemp /tmp/ocr-XXXXXX.png)

# Capture a selected area of the screen
scrot -s -o "$TMP_IMG"

# Preprocess the image for better OCR accuracy
convert "$TMP_IMG" -colorspace Gray -normalize -deskew 40% -threshold 60% "$TMP_IMG"

# Perform OCR on the screenshot and copy the text to clipboard
tesseract "$TMP_IMG" stdout -l eng --dpi 150 | xclip -selection clipboard

# Remove the temporary image file
rm "$TMP_IMG"

