#!/bin/bash
# Launch the Streamlit UI
cd "$(dirname "$0")"
# Use the same python3 your terminal uses
python3 -m pip install --user streamlit sentence-transformers requests langdetect numpy
python3 -m streamlit run resonance_ui.py
