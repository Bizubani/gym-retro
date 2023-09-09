
import argparse

import retro

from utilities.keys_extractor import ButtonExtractor
from utilities.discretizer import GradiusDiscretizer, BattletoadsDiscretizer
import grimmx
gradius_actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['UP', 'B'], ['DOWN', 'B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B'], ['A']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='GradiusIII-Snes')
    parser.add_argument('--state', default=retro.State.DEFAULT)
    parser.add_argument('--scenario', default=None)
    args = parser.parse_args()

    extractor = ButtonExtractor('./storage/gradius/loaded_10mins/final_2__53700.0.bk2', gradius_actions)
    loaded_actions = extractor.get_actions_from_movie()

    ia = grimmx.grimm_runner(game=args.game, state=args.state, scenario=args.scenario, discretizer=GradiusDiscretizer, loaded_actions=loaded_actions)
    ia.run()


if __name__ == "__main__":
    main()
