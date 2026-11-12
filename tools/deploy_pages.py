"""Publish the viewer + Z-A and UNITE models to GitHub Pages (gh-pages branch) with GLBs on a Hugging Face dataset repo (CORS-enabled, free).

Small files (games.json, index.json, manifests, thumbnails, app bundle) go in the Pages site;
each model.glb is uploaded as `<game>__<id>__model.glb` to the HF dataset repo. Run from the repo root:
    python tools/deploy_pages.py            # publish to GitHub Pages (default, --pages)
    python tools/deploy_pages.py --local    # build + serve using the local out/ folder only (no upload/push)
    python tools/deploy_pages.py --skip-release --skip-push   # stage + build the Pages site only
"""
import argparse, json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out'
SITE = OUT / 'pages_site'          # out/ is git-ignored
GAMES = {'za': 'pokemon-za-models', 'unite': 'pokemon-unite-models'}   # game -> Hugging Face dataset repo under the logged-in user (`hf auth login`)


def rmtree(path):
    """shutil.rmtree that also removes read-only files (git objects on Windows)."""
    def onexc(func, p, _):
        os.chmod(p, 0o700); func(p)
    shutil.rmtree(path, onexc=onexc) if Path(path).exists() else None


def run(*a, cwd=ROOT, env=None, check=True, capture=False):
    r = subprocess.run(a, cwd=cwd, env=env, check=check, text=True, capture_output=capture, shell=(os.name == 'nt' and a[0] == 'npm'))
    return r.stdout if capture else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', action='store_true', help='build for root path and serve models from local out/ (npm run preview); nothing is uploaded')
    ap.add_argument('--skip-release', action='store_true')
    ap.add_argument('--skip-push', action='store_true')
    args = ap.parse_args()

    if args.local:
        env = {k: v for k, v in os.environ.items() if k not in ('VITE_BASE', 'VITE_GLB_BASE')}
        run('npm', 'run', 'build', cwd=ROOT / 'viewer', env=env)
        print('serving local out/ at http://localhost:4173 (Ctrl+C to stop)')
        run('npm', 'run', 'preview', cwd=ROOT / 'viewer', env=env)
        return

    remote = run('git', 'remote', 'get-url', 'origin', capture=True).strip()
    owner, repo = re.search(r'github\.com[:/]([^/]+)/(.+?)(?:\.git)?$', remote).groups()

    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()['name']
    hf_ids = {g: f'{user}/{r}' for g, r in GAMES.items()}
    glb_bases = {g: f'https://huggingface.co/datasets/{h}/resolve/main' for g, h in hf_ids.items()}

    # 1. stage small model files
    rmtree(SITE)
    models = SITE / 'models'
    models.mkdir(parents=True)
    all_games = json.loads((OUT / 'games.json').read_text(encoding='utf-8'))
    (models / 'games.json').write_text(json.dumps({g: all_games[g] for g in GAMES}, ensure_ascii=False), encoding='utf-8')
    glbs = {g: {} for g in GAMES}
    for GAME in GAMES:
        (models / GAME).mkdir()
        index = json.loads((OUT / GAME / 'index.json').read_text(encoding='utf-8'))
        shutil.copy(OUT / GAME / 'index.json', models / GAME / 'index.json')
        for i in [m['id'] for m in index['models']]:
            src = OUT / GAME / i
            dst = models / GAME / i
            dst.mkdir()
            for name in ('manifest.json', 'thumb.webp'):
                if (src / name).exists():
                    shutil.copy(src / name, dst / name)
            glbs[GAME][f'{GAME}__{i}__model.glb'] = src / 'model.glb'
    big = [n for d in glbs.values() for n, p in d.items() if p.stat().st_size > 2_000_000_000]
    if big:
        sys.exit(f'GLBs over the 2 GB release-asset limit: {big}')
    site_mb = sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file()) / 1e6
    print(f'{sum(len(d) for d in glbs.values())} models: site models/ {site_mb:.0f} MB')

    # 2. build the viewer
    env = dict(os.environ, VITE_BASE=f'/{repo}/', VITE_GLB_BASES=json.dumps(glb_bases))
    run('npm', 'run', 'build', cwd=ROOT / 'viewer', env=env)
    dist = ROOT / 'viewer' / 'dist'
    for item in dist.iterdir():
        if item.is_dir():
            shutil.copytree(item, SITE / item.name, dirs_exist_ok=True)
        else:
            shutil.copy(item, SITE / item.name)
    shutil.copy(SITE / 'index.html', SITE / '404.html')   # SPA deep links
    (SITE / '.nojekyll').write_text('')
    total_mb = sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file()) / 1e6
    print(f'site total {total_mb:.0f} MB (Pages limit ~1000 MB)')

    # 3. GLBs -> Hugging Face datasets (hardlinks under the asset names; unchanged files are skipped by hash)
    if not args.skip_release:
        for g, files in glbs.items():
            api.create_repo(hf_ids[g], repo_type='dataset', exist_ok=True)
            stage = OUT / f'pages_release_{g}'
            rmtree(stage); stage.mkdir()
            for n, p in files.items():
                try: os.link(p, stage / n)
                except OSError: shutil.copy(p, stage / n)
            api.upload_folder(repo_id=hf_ids[g], repo_type='dataset', folder_path=str(stage), commit_message=f'{g} GLBs')
            print(f'uploaded {len(files)} GLBs to https://huggingface.co/datasets/{hf_ids[g]}')

    # 4. push site to gh-pages (separate repo in out/pages_site; main is untouched)
    if not args.skip_push:
        run('git', 'init', '-q', '-b', 'gh-pages', cwd=SITE)
        run('git', 'add', '-A', '-f', cwd=SITE)
        run('git', 'commit', '-q', '-m', 'Deploy viewer', cwd=SITE)
        run('git', 'push', '-f', remote, 'gh-pages', cwd=SITE)
        print(f'pushed; enable Pages (source gh-pages) once: https://github.com/{owner}/{repo}/settings/pages')
        print(f'site: https://{owner}.github.io/{repo}/')


if __name__ == '__main__':
    main()
