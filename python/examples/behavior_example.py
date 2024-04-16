import pathlib
import os

import webpdfparse

# Get the absolute path
CWD = pathlib.Path(os.path.abspath(__file__)).parent

webpdfparse.analyze_pdf(CWD / "behavior_mummy-1.pdf")