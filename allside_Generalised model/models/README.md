# models/

Put the trained surrogate here:

    best_facade_allsides_climate_genroom.pth    (120 MB, from the v1.0 GitHub release)
    SHA-256 0be1a5710ffbe7cec4b90ae3fe4e598b42892252d43c081270a8bb3940b30e61

It is a bare `state_dict` for `AllSidesClimateTransformer(**MODEL_CONFIG)` in
`model trainer/model_allsides_climate.py` (31,181,572 parameters), and it is too large for git.

Training (`model_allsides_climate.py` run as a script) also writes
`checkpoint_allsides_climate_genroom.pth` here and **overwrites** the best-weights file. Keep a
copy of the release weights before retraining.
