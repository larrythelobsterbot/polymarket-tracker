"""
Build script to create a standalone executable for Polymarket Tracker.
Run this script to generate PolymarketTracker.exe
"""

import PyInstaller.__main__
import os
import sys

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the main entry point (use the wrapper script)
main_script = os.path.join(script_dir, "run_gui.py")

# PyInstaller arguments
args = [
    main_script,
    "--name=PolymarketTracker",
    "--onefile",  # Single executable file
    "--windowed",  # No console window (GUI app)
    "--noconfirm",  # Overwrite without asking
    f"--distpath={os.path.join(script_dir, 'dist')}",
    f"--workpath={os.path.join(script_dir, 'build')}",
    f"--specpath={script_dir}",
    # Hidden imports that PyInstaller might miss
    "--hidden-import=polymarket_tracker",
    "--hidden-import=polymarket_tracker.database",
    "--hidden-import=polymarket_tracker.config",
    "--hidden-import=polymarket_tracker.analytics",
    "--hidden-import=polymarket_tracker.collector",
    "--hidden-import=polymarket_tracker.api_client",
    "--hidden-import=polymarket_tracker.win_rate",
    "--hidden-import=polymarket_tracker.insider_detection",
    "--hidden-import=polymarket_tracker.gui",
    "--hidden-import=aiohttp",
    "--hidden-import=pydantic",
    "--hidden-import=pydantic_settings",
    "--hidden-import=dotenv",
    "--hidden-import=tkinter",
    "--hidden-import=tkinter.ttk",
    "--collect-all=pydantic",
    "--collect-all=pydantic_settings",
    # Collect the entire polymarket_tracker package
    "--collect-all=polymarket_tracker",
]

if __name__ == "__main__":
    print("Building Polymarket Tracker executable...")
    print("This may take a few minutes...")
    PyInstaller.__main__.run(args)
    print("\nBuild complete!")
    print(f"Executable created at: {os.path.join(script_dir, 'dist', 'PolymarketTracker.exe')}")
