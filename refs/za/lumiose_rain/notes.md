# lumiose_rain: notes

Status: no usable frames. Nothing was grabbed, no regions files were written, no fit was run (fit.json not produced).

## What was scanned

Search (`search --scene lumiose_rain --query "Pokemon Legends Z-A rain"`, and "Lumiose rain Sliggoo Goodra evolution")
returned mostly music/ambience uploads and X/Y or other-game videos. Z-A candidates checked:

- `xj5EyVEzcH4` (Let It Rain, Let It Pour side quest, 143 s, 0-150 s scouted): clear blue daytime sky, no rain.
- `2AiS8_FTDhA` (How to change weather / Goodra, 99 s, 0-100 s scouted): clear sky, no rain.
- `1HFTURzKj2Q` (How to get Goodra, 113 s, 0-100 s scouted): clear sky, no rain.
- `tdNSQbZuor4` (How to make it rainy, 85 s, 0-80 s scouted): no rain visible; a Battle Zone clip only.
- `I6vC54mTxpY` (1 h "Rainy Day in Lumiose City" music video, 0-10 min scouted at 30 s steps): a fogged, grey
  rooftop view with no characters or HUD. Lighting is interesting for the scene but there is no catalogued subject,
  so it cannot be fitted. Not used.

Full game `6qiJ2FnPSFM` (Shirrako, 18.6 h, 66936 s): 19 sampled 10-minute windows at 30 s steps, one per hour
(start times 0:20, 1:20, 2:20 ... 18:20; 0:20 window read at 1200-1800 s). Day, dusk and night Lumiose, Battle Zone,
Wild Zones and interiors were seen. No rain was visible in any sample. Some frames have grey or overcast skies
(e.g. ~3:26, 13:25) but no wet streets, rain streaks or grey rain look. Wild Zone 12 at 8:20 has snow patches,
not rain.

Coverage: about 3.2 of 18.6 hours of the full game were looked at (sparse sampling). Rain may still exist in
the unsampled parts (the Goodra quest and the Sliggoo evolution need rain), so a denser scan (every 10 minutes over
the full game) would be the next step.

## Why it stopped

Per the brief: no rain frame with a catalogued subject was found, so the scene stops here. The scout sheets are in
`refs/za/lumiose_rain/scout/` (git-ignored PNGs; `search.json` and `6qiJ2FnPSFM_*_s00.png` list the sampled windows).

## Doubts

- Sparse sample: a rain frame may exist in the 15 h not sampled.
- The Goodra and Sliggoo evolution videos found in search are X/Y (Kalos) uploads, not Z-A, so they are useless.
- The fog and grey rooftop in the music video is a possible weather reference, but it has no subject to fit.
