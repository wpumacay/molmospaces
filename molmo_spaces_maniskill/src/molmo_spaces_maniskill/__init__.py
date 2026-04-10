from pathlib import Path

# Points to src/molmo_spaces_maniskill/
MOLMOSPACES_MANISKILL_BASE_DIR = Path(__file__).parent

# Points to the project root (molmo_spaces_maniskill/)
# This is where assets/ directory is located
MOLMOSPACES_MANISKILL_ROOT = Path(__file__).parent.parent.parent

# Common asset paths
MOLMOSPACES_ASSETS_DIR = MOLMOSPACES_MANISKILL_ROOT / "assets"
MOLMOSPACES_MJCF_SCENES_DIR = MOLMOSPACES_ASSETS_DIR / "mjcf" / "scenes"
