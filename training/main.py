
import argparse
import retro

from utilities.keys_extractor import ButtonExtractor
from utilities.discretizer import GradiusDiscretizer, BattletoadsDiscretizer
import grimmx
gradius_actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['UP', 'B'], ['DOWN', 'B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B'], ['A']]
battletoad_and_double_dragon_actions = [['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['Y'], ['B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B', 'Y'], ['LEFT', 'Y'], ['RIGHT', 'Y']]

def main():
  
    # set up the argument parser and parse the arguments   
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='GradiusIII-Snes')
    parser.add_argument('--state', default=retro.State.DEFAULT)
    parser.add_argument('--scenario', default=None)
    parser.add_argument('--record_path', default='./storage/')
    parser.add_argument('--file_to_load', default=None)
    parser.add_argument('--penalty_scale', default=1000)
    parser.add_argument('--episode_steps', default=4500)
    parser.add_argument('--time_step_limit', default=1e7)
    args = parser.parse_args()
    if args.game == 'gradius':
        game = 'GradiusIII-Snes'
        discretizer = GradiusDiscretizer
        action_set = gradius_actions
    elif args.game == 'battletoads':
        game = 'BattletoadsDoubleDragon-Snes'
        discretizer = BattletoadsDiscretizer
        action_set = battletoad_and_double_dragon_actions
    elif args.game == 'final_fight':
        game = 'FinalFight2-Snes'
    elif args.game == 'double_dragon':
        game = 'SuperDoubleDragon-Snes'
    else:
        print('\x1B[3mGame not recognized\x1B[0m')
        exit(1)
    if args.file_to_load is None:
        print('\x1B[3mNo file specified. Playing from scratch.\x1B[0m')
        loaded_actions = None
    else:
        extractor = ButtonExtractor(args.file_to_load, action_set)
        loaded_actions = extractor.get_actions_from_movie()
    #call the agent and have it set up itself
    ia = grimmx.grimm_runner(game=game, state=args.state, scenario=args.scenario,
                              discretizer=discretizer, record_path=args.record_path,
                              loaded_actions=loaded_actions, penalty_scale_arg=int(args.penalty_scale),
                                max_episode_steps=int(args.episode_steps), timestep_limit=int(float(args.time_step_limit)
                              ))
    print('Agent set up. Ready to run.')
    ia.run()



if __name__ == "__main__":
    main()
