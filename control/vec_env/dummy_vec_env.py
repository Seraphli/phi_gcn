import numpy as np
from . import VecEnv

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        # Handle gymnasium format: obs, reward, terminated, truncated, info
        obs, rews, terminateds, truncateds, infos = map(np.array, zip(*results))
        dones = terminateds | truncateds  # Combine terminated and truncated
        self.ts += 1
        for (i, done) in enumerate(dones):
            if done:
                # Try gymnasium format first (with seed parameter)
                if hasattr(self.envs[i], '_seed'):
                    try:
                        result = self.envs[i].reset(seed=self.envs[i]._seed)
                        obs[i] = result[0] if isinstance(result, tuple) else result
                    except TypeError:
                        # Fallback to old gym format
                        result = self.envs[i].reset()
                        obs[i] = result[0] if isinstance(result, tuple) else result
                else:
                    result = self.envs[i].reset()
                    obs[i] = result[0] if isinstance(result, tuple) else result
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = []
        for env in self.envs:
            # Try gymnasium format first (with seed parameter)
            if hasattr(env, '_seed'):
                try:
                    result = env.reset(seed=env._seed)
                except TypeError:
                    # Fallback to old gym format
                    result = env.reset()
            else:
                result = env.reset()
            results.append(result)
        
        # Handle gymnasium reset format: obs, info
        obs = [result[0] if isinstance(result, tuple) else result for result in results]
        return np.array(obs)

    def close(self):
        return