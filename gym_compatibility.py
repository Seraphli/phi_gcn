"""
Compatibility layer for migrating from gym to gymnasium.
This file provides backward compatibility for the old gym API.
"""

try:
    # Try to import gymnasium first (modern version)
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.core import Wrapper, ObservationWrapper, RewardWrapper
    from gymnasium.spaces.box import Box
    
    # For backward compatibility, create aliases
    gym.Wrapper = Wrapper
    gym.ObservationWrapper = ObservationWrapper  
    gym.RewardWrapper = RewardWrapper
    gym.spaces = spaces
    
    print("Using gymnasium (modern gym)")
    
except ImportError:
    # Fallback to old gym if gymnasium is not available
    try:
        import gym
        from gym import spaces
        from gym.core import Wrapper, ObservationWrapper, RewardWrapper
        from gym.spaces.box import Box
        
        print("Using legacy gym")
        
    except ImportError:
        raise ImportError("Neither gymnasium nor gym is installed. Please install gymnasium: pip install gymnasium[atari]")

# Export the gym module for use in other files
__all__ = ['gym', 'spaces', 'Wrapper', 'ObservationWrapper', 'RewardWrapper', 'Box']