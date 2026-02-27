"""
Entry point for Polymarket Tracker GUI.
This script handles imports properly for PyInstaller.
"""

import sys
import os

# Add the parent directory to path for imports
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    application_path = sys._MEIPASS
    # Also set working directory to where the exe is located
    os.chdir(os.path.dirname(sys.executable))
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, application_path)

# Now import and run the GUI
from polymarket_tracker.gui import main

if __name__ == "__main__":
    main()
