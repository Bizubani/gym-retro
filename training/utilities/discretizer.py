import gym
import numpy as np

gradius_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["UP", "B"],
    ["DOWN", "B"],
    ["LEFT", "B"],
    ["RIGHT", "B"],
    ["B"],
    ["A"],
]
battletoad_and_double_dragon_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["Y"],
    ["B"],
    ["LEFT", "B"],
    ["RIGHT", "B"],
    ["B", "Y"],
]
rtype_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["A", "B"],
    ["Y"],
    ["X"],
    ["LEFT", "X"],
    ["RIGHT", "X"],
    ["UP", "X"],
    ["DOWN", "X"],
]
draculax_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["Y"],
    ["B"],
    ["X"],
    ["LEFT", "B"],
    ["RIGHT", "B"],
]
final_fight_2_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["Y"],
    ["LEFT", "B"],
    ["RIGHT", "B"],
    ["B"],
]

double_dragon_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["Y"],
    ["B"],
    ["LEFT", "B"],
    ["RIGHT", "B"],
    ["X"],
    ["A"],
    ["L"],
]

snes_actions = [
    ["LEFT"],
    ["RIGHT"],
    ["UP"],
    ["DOWN"],
    ["Y"],
    ["B"],
    ["A"],
    ["X"],
    ["L"],
    ["R"],
]


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        # removed assertion to allow compatibility with gym > 26
        # assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        print(f"\x1B[32mAction space size is {env.action_space.n}\x1B[0m")
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class GradiusDiscretizer(Discretizer):
    """
    Use Gradius-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=gradius_actions)


class BattletoadsDiscretizer(Discretizer):
    """
    Use Battletoads and Double Dragon-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=battletoad_and_double_dragon_actions)


class RTypeDiscretizer(Discretizer):
    """
    Use Rtype-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=rtype_actions)


class DraculaXDiscretizer(Discretizer):
    """
    Use DraculaX-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=draculax_actions)


class FinalFight2Discretizer(Discretizer):
    """
    Use FinalFight2-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=final_fight_2_actions)


class DoubleDragonDiscretizer(Discretizer):
    """
    Use DoubleDragon-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=double_dragon_actions)


# provide a unified interface for all snes games
class SnesDiscretizer(Discretizer):
    """
    Use Snes-specific discrete actions
    based on ttps://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    def __init__(self, env):
        super().__init__(env=env, combos=snes_actions)
