import datetime
import os

from game_mechanics import WildTictactoeEnv, flatten_board, load_dictionary

EXAMPLE_STATE = flatten_board(WildTictactoeEnv().board)


def check_submission() -> None:
    """Checks a user submission is valid."""
    expected_output_type = tuple
    competitor_code_dir = os.path.dirname(os.path.realpath(__file__))
    main = [entry for entry in os.scandir(competitor_code_dir) if entry.name == "main.py"][0]

    assert main.is_file(), "main.py isn't a Python file!"

    file_name = main.name.split(".py")[0]

    pre_import_time = datetime.datetime.now()
    mod = __import__(f"{file_name}", fromlist=["None"])
    time_to_import = (datetime.datetime.now() - pre_import_time).total_seconds()

    # Check importing takes a reasonable amount of time
    assert time_to_import < 2, (
        f"Your main.py file took {time_to_import} seconds to import.\n"
        f"This is much longer than expected.\n"
        f"Please make sure it's not running anything (training, testing etc) outside the "
        f"if __name__ == '__main__': at the bottom of the file"
    )

    # Check the choose_move() function exists
    try:
        choose_move = getattr(mod, "choose_move")
    except AttributeError as e:
        raise Exception(f"No function 'choose_move()' found in file {file_name}.py") from e

    # Check there is a TEAM_NAME attribute
    try:
        team_name = getattr(mod, "TEAM_NAME")
    except AttributeError as e:
        raise Exception(f"No TEAM_NAME found in file {file_name}.py") from e

    # Check TEAM_NAME isn't empty
    if len(team_name) == 0:
        raise ValueError(f"TEAM_NAME is empty in file {file_name}.py")

    # Check TEAM_NAME isn't still 'Team Name'
    if team_name == "Team Name":
        raise ValueError(
            f"TEAM_NAME='Team Name' which is what it starts as - "
            f"please change this in file {file_name}.py to your team name!"
        )

    # Try loading value function
    try:
        value_fn_dict = load_dictionary(team_name)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Value dictionary file called 'dict_{team_name}.pkl' cannot be found! "
            f"Check the file exists & that the name matches."
        ) from e

    # Check function runs and output is as expected
    action = choose_move(EXAMPLE_STATE, value_fn_dict)
    assert isinstance(action, tuple), (
        f"Action output by `choose_move()` must be type {expected_output_type}, "
        f"but instead {action} of type {type(action)} was output."
    )
    assert isinstance(action[0], int), (
        f"The first element in the Tuple outputted by `choose_move()` must be type int but instead "
        f"{action[0]} of type {type(action[0])} was output."
    )

    assert isinstance(action[1], str), (
        f"The second element in the Tuple outputted by `choose_move()` must be type str but instead "
        f"{action[1]} of type {type(action[1])} was output."
    )

    print(
        "Congratulations! Your Repl is ready to submit :)\n\n"
        f"It'll be using value function file called 'dict_{team_name}.pkl'"
    )
