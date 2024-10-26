from dataclasses import dataclass

@dataclass
class SecurityConfig:
    """Configuration class for different security levels"""
    rounds: int
    memory_cost: int
    transform_iterations: int
    compression_level: int