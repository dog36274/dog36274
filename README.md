# The Desk

Five-panel cross-asset dashboard (Korea FX, US rates, gold & real yields, frontier, UK gilts).
`fetch_data.py` -> `data/*.json` -> static `index.html`, refreshed by GitHub Actions, served by GitHub Pages.

## One-time setup
1. Optional: a FRED key (fred.stlouisfed.org/docs/api/api_key.html). Without one, the script scrapes FRED's public CSV endpoint. A BoK ECOS key (ecos.bok.or.kr) is still needed for the Korea policy rate; without it the rest of the Korea panel works.
2. Repo Settings > Secrets > Actions: add `BOK_API_KEY` (plus optional `FRED_API_KEY`, `STOOQ_API_KEY`).
3. Settings > Pages: deploy from branch `main`, root.
4. Actions tab > "Update Desk Dashboard" > Run workflow. Check the log for `[ok]` / `[FAIL]` per panel.

## Scrapers
- FRED: official API if a key is set, else `fredgraph.csv`.
- DMO: gilts in issue (report D1A export) and current remit (currentremit.pdf) are scraped with sanity bounds; any field that fails falls back to `data/dmo_manual.json`. These two parsers were written without seeing the live pages, so check the workflow log for `DMO ...` warnings on the first run.

## Verify on the first live run
- Stooq: if it returns a non-CSV page, set `STOOQ_API_KEY`.
- BoE series codes (`BOE_SERIES`) and ONS CDIDs (`ONS_SERIES`, two unset) in `fetch_data.py`.
- Meeting dates in `meetings.json` (entered from memory).

## Hand-edited files
`positions.json`, `meetings.json`, `data/wgc_manual.json` (quarterly), `data/dmo_manual.json`.

## Preview offline
`python fetch_data.py --demo --out /tmp/demo`, serve the folder, open `index.html?data=/tmp/demo` (synthetic data).

## Open it anytime (single file, no server)
`python fetch_data.py && python build_single.py` writes `the-desk.html` (Chart.js and data inlined). Double-click it.
The GitHub workflow rebuilds and commits this file on every run, so you can also just download it from the repo.
