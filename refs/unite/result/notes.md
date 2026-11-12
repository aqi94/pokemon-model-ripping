# unite / result (unite_result)

Video: B-lxnC_eWNk (Ninbunchan, "Pokemon Unite Switch - Garchomp Gameplay & Online Match Win", switch1 1080p). Standard battle, Remoat Stadium.
Frames: 8:15.5 (result_garchomp_01, WIN screen with Garchomp + trainer, right-side coin panel) and 8:02.5 (result_garchomp_02, scoreboard, Garchomp left). Both Garchomp, default look, same pose.

Model kind: the results screen uses the LOBBY model (`unite/pm0445_00_lob`, clip `idle` frame 0): stance, proportions and arm pose match (dE 23 vs 31-33 for the battle model idle/score/scoreidle, whose arms/pose do not match). So results = lobby rig.
Boxes are manual (auto-regions rejects the shaded body): belly yellow, chest red, thigh purple. Overlays (skill icons, nameplate) avoided; trainer occludes the right side only.

Fit: joint mean dE 5.15 (studio seed start 9.76). Per shot: 01 3.9, 02 6.4 (belly yellow 12.8, shot is brighter/yellower than render).
Doubts: one Pokemon, three boxes each; the scene is a stylised backdrop with screen-space grading, so tone mapping/exposure are compensating. fetch_frames sidecar validation rejects UNITE ids (MODEL_RE), so sidecars use `pm0445_00_00` as subject.model; regions JSON uses the real `unite/...` id.
