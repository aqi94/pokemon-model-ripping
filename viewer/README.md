# Pokémon Model Viewer

Vite + React + TypeScript + three.js viewer for the models produced by `pipeline/` into `../out/`.

    npm install
    npm run dev        # http://localhost:5173  (serves /models/* from ../out/)
    npm run build      # type-check + production build into dist/
    npm run preview    # serves dist/ on :4173; /models/* is also mapped to ../out/ by the preview middleware

For a static deployment, copy or symlink `out/` to `dist/models` (e.g. `cp -r ../out dist/models`).

Routes: `/` browse grid, `/model-viewer?id=za/pm0001_00_00`, `/pokemon/pm0001_00_00`.
Keys: Left/Right previous/next clip, P pause, T time scale.
