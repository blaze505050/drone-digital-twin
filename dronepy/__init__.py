"""
dronepy
=======
Top-level forwarder for drone_sdk.dronepy.
Enables clean RocketPy-style notebook syntax:
    from dronepy import Drone, Environment, Flight, MonteCarlo
"""
import sys
from pathlib import Path

# Ensure sdk directory is in Python path if not installed
_sdk_path = str(Path(__file__).resolve().parent / "sdk")
if _sdk_path not in sys.path:
    sys.path.insert(0, _sdk_path)

from drone_sdk.dronepy import *  # noqa: F401, F403
from drone_sdk.dronepy import __all__, __version__, __author__
