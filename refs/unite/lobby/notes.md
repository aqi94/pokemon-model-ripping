# UNITE lobby

Video s2ra5myIAmc (AyDaGamer, "Pokemon Unite - Gameplay (No Commentary)", Switch, 1920x1080), 0:19.0, main lobby screen
(Halloween event stage, moonlit blue/cyan backdrop). Subject: Gardevoir, default look -> model unite/pm0282_00_lob, clip idle@0.
Sidecar uses model pm0282_00_00 because tools/lightrefs/sidecar.py MODEL_RE rejects UNITE ids; the regions JSON has the real id.

Fit: studio-seed baseline mean dE 11.73 (hand boxes) -> 5.09 joint fit (profile indoor, 3 passes, 2 starts).
Boxes: 3 white dress patches (left shadowed, right, lower centre). Green head box dropped (bonnet/face-mask overlap in the shot).

Doubts: single frame, only near-white dress boxes (no skin/colour regions), so saturation (2.0, at limit), ground (#59fffa)
and elevation 0 are poorly constrained and mostly absorb the scene's cyan Halloween event lighting, not UNITE's neutral lobby.
Lobby dress_mid_lower dE 10.7 (under-lit in shot by pumpkin glow). Other candidates tried: holowear comparison videos
(X9cLyMiaQXU, t5IE4VLIgPs, Osz-D9kt25A) are split-screen/tiny or battle footage; not usable. Needs a non-Halloween lobby frame.
