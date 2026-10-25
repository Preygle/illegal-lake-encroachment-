"""
dashboard/app.py
================
Lake Encroachment Monitor - two pages:

  Bengaluru lakes     the six study lakes, mapped by the U-Net (bengaluru.py)
  Any lake in India   search any lake, download its imagery, enhance it, map it
                      with the U-Net, and project encroachment (any_lake.py)

    run_dashboard.bat                          (double-click), or
    python -m streamlit run dashboard/app.py
"""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Lake Encroachment Monitor", page_icon="🛰️", layout="wide")
st.navigation([
    st.Page(HERE / "bengaluru.py", title="Bengaluru lakes", icon=":material/water:", default=True),
    st.Page(HERE / "any_lake.py", title="Any lake in India", icon=":material/travel_explore:"),
]).run()
