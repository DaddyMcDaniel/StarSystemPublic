chosen_system("unified_miniplanet.py").
chosen_command("commands/LOP_miniplanet").
chosen_viewer("renderer/pcc_game_viewer.py").

working_seed_file("unified_planet_seed_42.json").
working_seed_file("runs/miniplanet_seed_42.json").
working_seed_file("generated_planets/metadata/MiniWorld_20250821_213628.json").

deprecated_generator("create_mini_planet.py").
deprecated_generator("generate_miniplanet.py").

should_preserve(File) :- working_seed_file(File).
should_preserve(File) :- chosen_system(File).
should_preserve(File) :- chosen_command(File).
should_preserve(File) :- chosen_viewer(File).

should_remove(File) :- deprecated_generator(File).

::print "Files to preserve:".
::print should_preserve.
::print "Files to remove:".
::print should_remove.