"""
Centralized configuration for the IMC-PALM artifact.

All scripts import paths and shared experiment parameters from here so that the
repository is portable (no hard-coded absolute paths). Override the output
location with the IMC_PALM_DATA environment variable if desired:

    export IMC_PALM_DATA=/path/to/output
"""

import os

# ------------------------------------------------------------------
# Directory layout
# ------------------------------------------------------------------
# Root for all generated artifacts (task sets, CSV results, figures).
# Defaults to ./results inside the repository; override via env var.
RESULT_DIR = os.environ.get(
    "IMC_PALM_DATA",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)

# Generated synthetic task sets (JSON) live under RESULT_DIR/data.
DATA_DIR = os.path.join(RESULT_DIR, "data")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Shared experiment parameters (see paper Section 4.1)
# ------------------------------------------------------------------
M_VALUES = [2, 4, 8]
TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
P_H_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

# Task-set counts.
NUM_TESTS_EVAL = 5000     # offline schedulability (Section 4.2)
NUM_VALID_SIM = 100       # schedulable sets generated for simulation
MAX_SIM_SETS = 1000       # sets actually used per runtime point (Section 4.3)
SIM_TICKS = 10000         # simulation horizon in time units

# Default runtime parameters.
DEFAULT_SWITCH_PROB = 0.20    # P^MS
