"""Models for agent responses and signals."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class RayDalioSignal(BaseModel):
    """Ray Dalio investment signal model."""
    signal: str = Field(..., description="Investment signal: bullish, bearish, or neutral")
    confidence: float = Field(..., description="Confidence level from 0 to 100")
    reasoning: str = Field(..., description="Reasoning behind the investment signal")
