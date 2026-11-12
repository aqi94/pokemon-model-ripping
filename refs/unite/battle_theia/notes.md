# battle_theia (UNITE, Theia Sky Ruins day)

Frame used: video n11UvsJySKQ (I Am Ronak, mobile 1080p30, "Gengar Gameplay in New Map Theia Sky Ruins") at 2:14.6,
Gengar default look, model unite/pm0094_00, clip idle@0, yaw 15, high top-down camera (aligned by fit_scene).
Sidecar model is written as pm0094_00_00 because sidecar.MODEL_RE requires 3 groups (UNITE ids are pm####_##); regions JSON uses unite/pm0094_00.

Rejected: HzVZDCDWewM (Snorlax, wearing a holowear: black hoodie), JfVKk2DJ0Mo (Charizard, that is Auroma Park, not Theia),
Gengar 0:40 in n11UvsJySKQ (too small, effects).

Fit: studio seed baseline dE 19.98 -> day seed 0.69 (best). Only ONE frame and TWO boxes (lit head/body vs shadowed lower body,
mat_bat_0094body_lv3) so the fit is underdetermined: key/ambient split and backlight 2.7 are weakly constrained. Footage is
compressed mobile video with bloom/colour grading. Treat as a rough preset.
