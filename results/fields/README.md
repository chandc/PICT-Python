# Checkpoints live here and are not committed

`.npz` files in this directory are excluded by `.gitignore`. They are rewritten every run and
git would store each version whole, growing the repository without bound.

They are regenerable. Each production run's exact configuration is recorded in `reference/`, and
every driver takes its settings as arguments -- notably `--tol`, which is the one that decides
whether the square-cylinder case sheds at all.

What IS committed: `results/*.npy` (probe histories -- the measurements) and `results/logs/`
(what was run, and what it printed).
