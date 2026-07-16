"""Neural/learned-model subsystem for SpecForge.

Being built out per dev/NN_REPAIR_IMPLEMENTATION_PLAN_07.15.2026.md. The typed
NeuralForecast contract lives here so inference, scoring, the graph, audit, and
the UI all speak the same shape. The rest of neural.py migrates into this
package in a later stage; for now only the contract is here.
"""
from .schema import NeuralForecast

__all__ = ["NeuralForecast"]
