# Build Log

A running record of what was built, why, and what was decided. Newest phase at the bottom.

---

## Phase 0: Scaffolding + messy data generator (2026-10-27)

### Focus decision
The project centers on **patient matching**: identifying which raw records belong to the same real person. It's one of the hardest real problems in healthcare data, and it can be measured objectively against an answer key.

### What was built
- `src/generate_messy_data.py`: generates 500 true people, registers them across 3 source systems (`clinic_a`, `clinic_b`, `er_system`), creates visits with free-text notes, and injects mess. Writes raw CSVs plus an answer key.
- `src/inspect_raw.py`: DuckDB SQL profile showing the mess exists.
- README skeleton with the GCP mapping, `.gitignore`, `requirements.txt`.

### Decisions and tradeoffs
| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Table design | Two tables (patients, encounters) | One wide table | Realistic; enables orphan / foreign-key checks later |
| Fake names | Hand-written lists | `Faker` library | No black box; every value is visible in the code |
| Generated data in git | No, regenerate from seed | Commit CSVs | Repo stays small; same seed = identical files |
| "Today" | Fixed `EXTRACT_DATE = 2026-10-26` | `date.today()` | Reproducible: re-running next week gives the same data |
| Raw loading | Everything as text (`all_varchar`) | Let DuckDB guess types | Guessing silently destroys evidence (e.g. leading zeros in MRNs) |
| Phone numbers | 555-01XX range | Random numbers | Reserved for fiction, so they can't be real; shared numbers are a realistic matching trap |

### Mess injected (seed 42)
500 true people → **747** raw patient rows and **1,400** raw encounter rows.

| Mess type | Count | Why it's a problem |
|---|---|---|
| People registered in 2+ systems | 156 | Same person, different MRNs |
| Re-registered in same system, new MRN | 25 | Can't trust "one MRN per person" even within one system |
| Nickname used (Robert→Bob) | 25 | Exact name match fails |
| Last-name typo (Smith→Smtih) | 29 | Exact name match fails |
| Twin pairs | 7 | Same last name, DOB, phone, ZIP but **different people**; over-eager matching merges them |
| Household members sharing a phone | 25 | Phone alone can't identify a person |
| Two-digit-year DOBs (`3/4/52`) | 78 | Python reads `52` as **2052** by default |
| Impossible DOBs (future / before 1900) | 7 | Typos that pass a format check |
| Missing values across 6 spellings | 243 in patients | `""`, `N/A`, `NULL`, `unknown`, `none`, `-` all mean "unknown" |
| Exact duplicate rows | 24 patients / 39 encounters | Batch job loaded rows twice |
| MRN lost leading zeros (Excel) | 54 | Encounter looks orphaned but is fixable |
| True orphan encounters | 26 | Points at a patient that doesn't exist |
| Impossible vitals (BP 999, 0, -1) | 26 | Corrupts averages and model features |
| Discharge before admit | 31 | Negative length of stay |
| PHI leaked into notes | 68 | Phone/name/MRN in free text; column masking won't catch it |
| Empty notes | 44 | No text to label from |

Full counts: `data/answer_key/mess_manifest.json`.

### Things noticed while profiling
- A crude match on "last name + first initial" flags 186 groups. Many are real duplicates, but some lump together different people (James, Jennifer and Joseph Wilson). That's exactly why Phase 1 needs proper matching rules.
- Of 68 clinic_b orphan encounters, 58 are fixable by zero-padding the MRN back to 8 digits. The other 10 are true orphans.
- DuckDB turns an empty field into a real `NULL` even when loading as text, while the typed string `"NULL"` stays text. The two must be handled differently.

### Open questions for Phase 1
- Which fields should a match require, and which only add confidence?
- What happens to records we can't confidently match: merge, keep separate, or send to human review?

---

## Phase 1: Ingestion, cleaning, patient matching, quarantine (2026-10-27)

### What was built
- `src/clean.py`: one small function per repair (missing tokens, 5 date formats, phone, ZIP, sex, payer names, vitals, MRN zero-padding).
- `src/match_patients.py`: blocking → field-by-field scoring with vetoes → auto_match / review / non_match → clusters.
- `src/pipeline.py`: raw → `data/clean/` (patient records, golden patients, encounters), `data/quarantine/`, `data/review/match_review_queue.csv`.
- `src/evaluate_phase1.py`: grades matching and encounter linkage against the answer key, next to two naive baselines.
- `tests/test_phase1.py`: 13 pytest tests, including twins, households, nicknames, typos and the two-digit-year trap.

### Answers to the Phase 0 open questions
**Which fields must a match have?** First name (after nickname → formal name) and exact DOB, for an *automatic* merge. Phone and ZIP only add confidence, because twins and households share them. Some disagreements are vetoes: different first names, different sex, or DOBs that don't even share a month and day.

**What happens to uncertain records?** They stay separate and go to a human review queue. A wrong merge mixes two people's medical histories (wrong allergies, wrong meds), which is far worse than a leftover duplicate. So the rules trade some recall for zero wrong merges.

### Decisions and tradeoffs
| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Matching method | Hand-written rules with points and vetoes | ML model / Splink-style probabilistic weights | Every decision is explainable in a review queue; no training labels needed |
| Auto-merge rule | Requires first name AND exact DOB | Score threshold only | A high score can still be a twin with a blank first name |
| Uncertain pairs | Human review queue, not merged | Merge above a lower threshold | Wrong merges are the dangerous error in healthcare |
| Candidate pairs | Blocking on DOB, phone, first name + last initial | Compare every pair | 1,247 comparisons instead of 261,003; the only way this works at millions of rows |
| Last-name typos | Allow one edit or neighbour swap | Fuzzy score (Jaro-Winkler) | Easy to explain: "one typo apart" |
| Two-digit years | Pivot on extract date (`'52` → 1952, `'22` → 2022) | Python default (`'52` → 2052) | Python's default breaks every DOB before 1969 |
| Impossible DOB | Set to NULL and flag | Quarantine the whole patient | The person is real; only one field is wrong |
| Impossible vitals | Set to NULL and flag | Quarantine the visit | The visit happened; one bad number shouldn't hide it |
| Discharge before admit | Quarantine | Swap the dates | Probably swapped, but we can't know the true dates |
| MRN lost leading zeros | Pad to 8 digits, only if the padded MRN exists | Pad always / give up | A repair is only safe if it's confirmed |
| Golden record | Most common name/DOB; most recent contact info | Most recent everything | A one-off typo gets outvoted |

### Results (seed 42)
| Method | Pairs merged | Wrong merges | Missed | Precision | Recall |
|---|---|---|---|---|---|
| Exact match on raw text | 26 | 0 | 245 | 1.000 | 0.096 |
| Last name + DOB + phone | 202 | 8 | 77 | 0.960 | 0.716 |
| **Pipeline** | 246 | **0** | 25 | **1.000** | **0.908** |

- 723 clean patient records → **519 golden patients** (500 true people). The 19 extra are the 25 missed pairs, all of them in the review queue.
- Review queue: 31 pairs, 25 truly the same person. One of the 6 "no" pairs is the twin trap working: a Jackson record with a blank first name matches twin Jennifer Jackson on everything else.
- Encounters: 1,400 raw → 39 exact duplicates dropped → 57 quarantined (31 discharge-before-admit, 26 orphan MRN) → 1,304 clean, **all linked to the correct person**. All 54 lost-leading-zero MRNs were repaired.
- Checked on seeds 7 and 2026 so the rules aren't tuned to one dataset: 0 wrong merges both times, recall 0.94 and 0.91.

### Things noticed while building
- **Bug caught by the evaluation:** pandas stores a missing value as `NaN`, and `NaN is not None`. A missing sex looked like a *different* sex, vetoed good matches and dropped recall to 0.73. Fixed by converting `NaN` → `None` before matching. Without the answer key, this would have shipped silently.
- 6 of the 7 impossible DOBs were caught. The 7th was a "-100 years" typo that landed in a plausible year (the 1910s). Range checks can't catch an error that looks normal.
- The nickname list in `clean.py` covers the nicknames this generator uses. Real data would need a much bigger public nickname list, so this is partly testing against known problems (see README limitations).
- 6 of the 31 review pairs are only "same first + last name, no DOB": weak evidence that wastes reviewer time. Raising the review threshold would drop them but also lose some true pairs. Left as is for now.

### Open questions for Phase 2
- Which columns are PHI, and who is allowed to see each one unmasked?
- How do we find PHI hidden in free-text notes (68 leaked phones, names and MRNs), not just in columns?
