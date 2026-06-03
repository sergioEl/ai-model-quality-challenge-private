from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Sampler(ABC):
    """Abstract base class for benchmark pruning samplers."""
    
    @abstractmethod
    def __call__(self) -> List[Dict[str, Any]]:
        """Load data, score samples, and return the pruned subset."""
        pass