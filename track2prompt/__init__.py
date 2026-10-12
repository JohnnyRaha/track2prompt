"""track2prompt - analyse audio you produced and generate music-model prompts."""
__version__ = "0.1.0"

from .analysis import load, analyse_rhythm, analyse_harmony, analyse_production
from .structure import analyse_structure
from .prompt import build_prompt, refine_prompt
from .suno import build_suno, format_suno

__all__ = ["load", "analyse_rhythm", "analyse_harmony", "analyse_production",
           "analyse_structure", "build_prompt", "refine_prompt",
           "build_suno", "format_suno"]
