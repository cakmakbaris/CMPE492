from pydantic import Field
from src.schemas import StrictModel


class Settings(StrictModel):
    # Temporary engineering bounds, NOT clinically validated stopping criteria.
    max_rounds: int = Field(default=5, ge=1, le=100)
    max_new_tokens: int = Field(default=3072, ge=64)
    max_input_tokens: int = Field(default=16384, ge=256)
