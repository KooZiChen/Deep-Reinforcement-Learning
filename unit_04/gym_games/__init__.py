import gymnasium
from gymnasium.envs.registration import register
import gym.spaces as old_spaces
import gymnasium.spaces as new_spaces
import gym_pygame.envs as _gym_pygame_envs


def _convert_space(space):
    """Convert a gym space to a gymnasium space."""
    if isinstance(space, old_spaces.Discrete):
        return new_spaces.Discrete(space.n)
    if isinstance(space, old_spaces.Box):
        return new_spaces.Box(
            low=space.low, high=space.high, shape=space.shape, dtype=space.dtype
        )
    return space  # fallback: hope it's already compatible


class _GymToGymnasium(gymnasium.Env):
    """Wraps an old gym-style env to the gymnasium API."""

    def __init__(self, old_env_class, **kwargs):
        self.render_mode = kwargs.pop('render_mode', None)
        self._env = old_env_class(**kwargs)
        self.observation_space = _convert_space(self._env.observation_space)
        self.action_space = _convert_space(self._env.action_space)
        self.metadata = getattr(self._env, 'metadata', {})
        self.metadata['render_modes'] = ['rgb_array', 'human']

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info

    def render(self):
        if self.render_mode is None:
            return None
        return self._env.render(mode=self.render_mode)

    def close(self):
        return self._env.close()


_GAMES = {
    'Catcher':      _gym_pygame_envs.CatcherEnv,
    'FlappyBird':   _gym_pygame_envs.FlappyBirdEnv,
    'Pixelcopter':  _gym_pygame_envs.PixelcopterEnv,
    'PuckWorld':    _gym_pygame_envs.PuckWorldEnv,
    'Pong':         _gym_pygame_envs.PongEnv,
}

for _game, _cls in _GAMES.items():
    _wrapped_cls = type(
        f'{_game}GymnasiumEnv',
        (_GymToGymnasium,),
        {'__init__': lambda self, _c=_cls, **kw: _GymToGymnasium.__init__(self, _c, **kw)},
    )
    register(
        id=f'{_game}-PLE-v0',
        entry_point=_wrapped_cls,
    )
