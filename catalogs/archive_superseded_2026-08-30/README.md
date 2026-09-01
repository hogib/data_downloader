# Superseded catalogues — archived 2026-08-30

Moved out of the working path because scripts silently disagreed about which
catalogue to read, and the two here are materially incomplete.

Measured against AFAD's live API (BODT, 400 km, M>=2.5, 2024-05 -> 2026-08):

| file | events | note |
|---|---|---|
| `deprem_katalog_utc.csv` | 3,450 | produced the chaos forecasting null (233 events in the Q1 window -> 1,092 positives, matching `logs/chaos_screen.log`) |
| `extracted_earthquakes.csv` | 3,450 in-window | the detection **download list**; dataset provenance is duplicated in each `dataset_*/manifest.csv` `event_id` column |
| `pilot_continuous_sample.csv` | 15 rows | one-off pilot sample |
| *(kept in place)* `data_large.csv` | 4,082 | best local copy, still missing 29.3% vs AFAD |

All three are AFAD data despite the repo calling them KOERI: 100% of their
EventIDs are AFAD eventIDs, magnitudes and coordinates identical to the API.

**Retained, not deleted,** because these are the provenance record for results
already published in `model_cnn_lstm/docs/report.md` and
`docs/REPRODUCE_ponly.md`. Delete once those are re-derived on the replacement
catalogue.

---

## `data_large.csv` — archived the same day

Superseded by `../catalog_afad_2026-08-30.csv`, which is a **strict superset**
above M1.5: all 394,432 of `data_large`'s M>=1.5 events are present, plus 19,353
it lacked. It restores the February 2025 Aegean swarm (1,692 "Ege Denizi" rows
vs 153) and runs 17 days further.

Retained only because it is the sole local source of events **below M1.5**
(195,551 rows). Nothing in the repo uses those: forecasting thresholds at
M>=2.5, and the detection download list bottoms out at M2.0. Re-fetch with a
lower `--min-magnitude` rather than reaching back into this file.
