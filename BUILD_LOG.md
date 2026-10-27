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
