# SpatialAIM online demo — deployment

The GUI started with `--demo_data`: a read-only browser over precomputed
results of the public datasets.
Visitors pick a pair and explore it; there are no uploads, no
runs, no exports and no writes to disk (cell-type names live in the session only).
Plotly.js is served by the container itself and Streamlit's usage statistics are off.

```
deploy/demo/
├── Dockerfile        # image: spatial-aim + local Plotly.js, runs `spatialaim gui --demo_data /data`
└── legal/            # imprint.md, privacy.md
```

## 1. Data layout on the server

```
/srv/spatialaim-demo/data/        # mounted read-only at /data
├── datasets/                     # = Data/01_Datasets_Paper (scRNA/, ST/, pairs.csv, index.csv files)
└── results/
    ├── average/
    │   └── <PairID:03d>_<scName>__<stName>/   # one pair output each (spatialaim run --pairs_csv naming)
    └── ward/
        └── <PairID:03d>_<scName>__<stName>/
```

Each linkage folder is one batch run with that `--linkage_method`, e.g.

```bash
spatialaim run --pairs_csv datasets/pairs.csv --sc_dir datasets/scRNA --st_dir datasets/ST --output_dir results/ward --linkage_method ward --mapping <mapper> --annotation <obs_column> ...
```

Each pair output has the usual layout `<start>/<mapper>/` plus `reference_scaffold.h5ad`, as
written by `spatialaim run` or the GUI. Only pairs with at least one finished start (in any
linkage) appear in the picker, so the demo can go live with a subset and grow; the sidebar's
linkage dropdown offers the linkages a pair has. A pair output directly in `results/` (no linkage
folder) still works, under the linkage its configs record. An optional `Name`
column in `pairs.csv` sets the picker's label for a pair; empty or missing, it shows
`<PairID> · <scName> × <stName>`.

Results can be computed anywhere and copied over: the demo compares the scaffold cache key without
the input files' paths and modification times. It never rebuilds a missing scaffold (that pair then
shows a warning on the reference plots).

## 2. Get the image

GitHub Actions (`.github/workflows/docker-demo.yml`) builds it and pushes it to GHCR on every
published GitHub Release (tags `<version>` and `latest`) and on a manual run (tag
`sha-<commit>`); pull requests touching the image only build it.

```bash
docker pull ghcr.io/domipysch/spatialaim-demo:latest
```

```bash
docker build -f deploy/demo/Dockerfile -t ghcr.io/domipysch/spatialaim-demo:latest --build-arg SPATIALAIM_VERSION=1.4.0 .
```

The build downloads Plotly.js 2.35.2 from jsDelivr once; the running container needs no
internet access.

## 3. Legal texts

`legal/imprint.md` and `legal/privacy.md` are copied into the image, so they ship with the code:
edit them in the repository, and a change goes live with the next image (Release or manual
workflow run). The pages are linked from the sidebar footer on every page (`/imprint`, `/privacy`).

## 4. Run

```bash
docker run -d --name spatialaim-demo --restart unless-stopped -p 127.0.0.1:8501:8501 -v /srv/spatialaim-demo/data:/data:ro --read-only --tmpfs /tmp --tmpfs /home/spatialaim --memory 16g --cpus 4 ghcr.io/domipysch/spatialaim-demo:latest
```

The container serves plain HTTP on port 8501; `127.0.0.1:` keeps it reachable only for the
reverse proxy on the same host (drop it if the proxy runs elsewhere). The proxy must forward
WebSockets (Streamlit's `/_stcore/stream`). If it keeps access logs, the privacy policy has to
say so.

Health: `docker ps` shows the status of the image's healthcheck on `/_stcore/health`.

## 5. Sizing and limits

- All visitors share one Python process. The reference scaffold (the largest object) is cached
  for at most 4 starts at once, the other caches are bounded too; a visitor switching to another
  reference loads its scaffold from disk (seconds for small references, longer for the large ones).
- Memory: plan for the largest reference's scaffold × 4 plus headroom; 16 GB is the starting
  point, watch `docker stats` and adjust `--memory`.
- Streamlit has no limit on concurrent sessions; rate limiting, if needed, belongs in the proxy.

## 6. Updating

After a release: `docker pull` the new image, then `docker rm -f spatialaim-demo` and run it again
(section 4). New results only need to be copied into `results/` — the pair list refreshes within
10 minutes.
