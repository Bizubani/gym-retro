import argparse
import os
import retro

from utilities.keys_extractor import ButtonExtractor
from utilities.discretizer import (
    GradiusDiscretizer,
    BattletoadsDiscretizer,
    RTypeDiscretizer,
    DraculaXDiscretizer,
    FinalFight2Discretizer,
    DoubleDragonDiscretizer,
)
import grimm_x as johan
import grimm_p.grimm_p as sebastian

# the directory of the script being run
script_dir = os.path.dirname(os.path.abspath(__file__))

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


def main():
    # add in the tweaked integration files
    retro.data.Integrations.add_custom_path(
        os.path.join(script_dir, "integration-tweaks")
    )
    print(
        "GradiusIII-Snes" in retro.data.list_games(inttype=retro.data.Integrations.ALL)
    )
    # set up the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="GradiusIII-Snes")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--record_path", default="./storage/")
    parser.add_argument("--file_to_load", default=None)
    parser.add_argument("--penalty_scale", default=1000)
    parser.add_argument("--episode_steps", default=4500)
    parser.add_argument("--time_step_limit", default=1e7)
    parser.add_argument("--grimm", default="johan")
    args = parser.parse_args()
    if args.game == "gradius":
        game = "GradiusIII-Snes"
        discretizer = GradiusDiscretizer
        action_set = gradius_actions
    elif args.game == "battletoads":
        game = "BattletoadsDoubleDragon-Snes"
        discretizer = BattletoadsDiscretizer
        action_set = battletoad_and_double_dragon_actions
    elif args.game == "final_fight2":
        game = "FinalFight2-Snes"
        discretizer = FinalFight2Discretizer
        action_set = final_fight_2_actions
    elif args.game == "double_dragon":
        game = "SuperDoubleDragon-Snes"
        discretizer = DoubleDragonDiscretizer
        action_set = double_dragon_actions
    elif args.game == "rtype":
        game = "RTypeIII-Snes"
        action_set = rtype_actions
        discretizer = RTypeDiscretizer
    elif args.game == "draculax":
        game = "CastlevaniaDraculaX-Snes"
        discretizer = DraculaXDiscretizer
        action_set = draculax_actions
    else:
        print("\x1B[3mGame not recognized\x1B[0m")
        exit(1)
    if args.file_to_load is None:
        print("\x1B[3mNo file specified. Playing from scratch.\x1B[0m")
        loaded_actions = None
    else:
        extractor = ButtonExtractor(args.file_to_load, action_set)
        loaded_actions = extractor.get_actions_from_movie()
    # call the agent and have it set up itself
    if args.grimm == "johan":
        johan.grimm_runner(
            game=game,
            state=args.state,
            scenario=args.scenario,
            discretizer=discretizer,
            record_path=args.record_path,
            loaded_actions=loaded_actions,
            penalty_scale_arg=int(args.penalty_scale),
            max_episode_steps=int(args.episode_steps),
            timestep_limit=int(float(args.time_step_limit)),
        )
    elif args.grimm == "sebastian":
        sebastian.grimm_runner(
            game=game,
            state=args.state,
            scenario=args.scenario,
            discretizer=discretizer,
            record_path=args.record_path,
            loaded_actions=loaded_actions,
            penalty_scale_arg=int(args.penalty_scale),
            max_episode_steps=int(args.episode_steps),
            timestep_limit=int(float(args.time_step_limit)),
        )
    else:
        print("\x1B[3mGrimm not recognized\x1B[0m")
        exit(1)


if __name__ == "__main__":
    main()
