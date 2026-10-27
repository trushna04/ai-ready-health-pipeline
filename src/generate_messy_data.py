"""
generate_messy_data.py  (Phase 0)

Creates intentionally MESSY synthetic healthcare data, the kind of raw export
a hospital network really produces when three systems are merged together.

Every value here is fake. No real patient data is ever used.

How it works, in four steps:
  1. Create "true" people (the ground truth, one row per real human).
  2. Register those people in three source systems. Some people end up
     registered more than once, with typos, nicknames, and different MRNs.
     This is the PATIENT MATCHING problem the project is built around.
  3. Create encounters (visits) with free-text notes, vitals, and dates.
  4. Inject more mess (missing values, bad dates, impossible values,
     double-loaded rows, orphan encounters) and write everything to CSV.

Outputs:
  data/raw/patients.csv        <- what the pipeline is allowed to read
  data/raw/encounters.csv      <- what the pipeline is allowed to read
  data/answer_key/*.csv/.json  <- ground truth for GRADING only.
                                  The pipeline must never read this folder.

Run:
  python src/generate_messy_data.py            (default seed 42)
  python src/generate_messy_data.py --seed 7   (a different, still repeatable dataset)
"""

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings. All "how messy" knobs live here so they are easy to find and explain.
# ---------------------------------------------------------------------------

# A fixed "today" so the same seed always produces the same files.
# (If we used date.today(), re-running tomorrow would change the data.)
EXTRACT_DATE = date(2026, 10, 26)

SOURCE_SYSTEMS = ["clinic_a", "clinic_b", "er_system"]

# How many systems each person is registered in: 1, 2, or 3.
SYSTEM_COUNT_WEIGHTS = {1: 0.70, 2: 0.22, 3: 0.08}

RATE_SAME_SYSTEM_REREGISTER = 0.04  # person re-registered in the SAME system with a NEW MRN
RATE_TWIN_PAIRS = 0.015             # twins: same last name, DOB, phone, address. Different people!
RATE_HOUSEHOLD = 0.05               # family members sharing a phone and last name

RATE_MISSING = 0.08         # chance an optional field is blank / "N/A" / "unknown"...
RATE_BAD_DOB = 0.01         # DOB in the future or before 1900
RATE_EXACT_DUP_ROW = 0.03   # a row loaded twice by a buggy batch job
RATE_BAD_VITALS = 0.015     # BP of 999, heart rate of 0...
RATE_DISCHARGE_BEFORE_ADMIT = 0.02
RATE_PHI_IN_NOTE = 0.04     # phone/name/MRN typed into a free-text note
RATE_EMPTY_NOTE = 0.03
RATE_ORPHAN_ENCOUNTER = 0.02     # encounter pointing at a patient that doesn't exist
RATE_MRN_LEADING_ZEROS_LOST = 0.10  # clinic_b MRN "00412345" became "412345" (Excel did it)

# Six different ways real systems say "we don't know".
MISSING_TOKENS = ["", "N/A", "NULL", "unknown", "none", "-"]

# ---------------------------------------------------------------------------
# Fake name pools. Hand-written so every value is visible and explainable.
# ---------------------------------------------------------------------------

FEMALE_NAMES = ["Elizabeth", "Katherine", "Jennifer", "Margaret", "Patricia", "Susan",
                "Priya", "Aisha", "Mei", "Sofia", "Fatima", "Hannah", "Olivia", "Grace",
                "Yuki", "Maria", "Anjali", "Rosa", "Emily", "Nicole"]
MALE_NAMES = ["Robert", "William", "Michael", "Christopher", "James", "Daniel",
              "Wei", "Carlos", "Jamal", "Arjun", "Luis", "Kevin", "Noah", "Omar",
              "Andre", "David", "Ravi", "Thomas", "Joseph", "Samuel"]
LAST_NAMES = ["Smith", "Johnson", "Patel", "Nguyen", "Garcia", "Williams", "Brown",
              "Shah", "Kim", "Martinez", "Davis", "Lopez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Lee", "Harris", "Clark",
              "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
              "Scott", "Hill", "Green", "Adams", "Baker", "Nelson", "Carter",
              "Mitchell", "Perez", "Roberts", "Turner", "Phillips"]

# Formal name -> common nickname. A matcher that compares names exactly
# will think "Robert Smith" and "Bob Smith" are two different people.
NICKNAMES = {"Robert": "Bob", "William": "Bill", "Michael": "Mike",
             "Christopher": "Chris", "James": "Jim", "Daniel": "Dan",
             "Elizabeth": "Liz", "Katherine": "Kate", "Jennifer": "Jen",
             "Margaret": "Peggy", "Patricia": "Pat", "Susan": "Sue",
             "Joseph": "Joe", "Samuel": "Sam", "Thomas": "Tom"}

AREA_CODES = ["610", "484", "215", "267", "302"]  # PA / DE
ZIP_CODES = ["19425", "19380", "19355", "19087", "19103", "19104",
             "19341", "19335", "19401", "19406", "19801"]

# Insurance: canonical value -> spellings different systems use for the same payer.
INSURANCE_VARIANTS = {
    "Medicare": ["Medicare", "MEDICARE", "medicare part b"],
    "Medicaid": ["Medicaid", "MEDICAID", "PA Medicaid"],
    "Aetna": ["Aetna", "AETNA", "aetna ppo"],
    "Blue Cross": ["Blue Cross", "BCBS", "bluecross", "Independence Blue Cross"],
    "UnitedHealthcare": ["UnitedHealthcare", "UHC", "United Health"],
    "Self-pay": ["Self-pay", "SELF PAY", "self"],
}

# ---------------------------------------------------------------------------
# Diagnosis note templates. The KEY of each list is the true category
# (the label the answer key stores). Notes use real clinical shorthand,
# and some contain NEGATIONS on purpose ("r/o MI", "denies chest pain")
# so a naive keyword search would label them wrong.
# ---------------------------------------------------------------------------

NOTE_TEMPLATES = {
    "hypertension": [
        "HTN f/u, BP check. Continue lisinopril 10mg.",
        "Elevated BP readings, pt reports HA. Hx of hypertension.",
        "hypertension - med refill",
        "No hx of diabetes. BP elevated today, start amlodipine.",
    ],
    "type2_diabetes": [
        "T2DM f/u, A1c 8.2, adjust metformin.",
        "Diabetes mellitus type 2, poorly controlled.",
        "DM2 - foot exam, pt c/o tingling feet.",
        "pt w/ diabetes, denies chest pain, here for insulin teaching",
    ],
    "asthma": [
        "Asthma exacerbation, wheezing, albuterol neb x2.",
        "pt w/ hx asthma, SOB after exercise.",
        "Denies chest pain. Here for asthma inhaler refill.",
    ],
    "chest_pain": [
        "pt c/o CP, r/o MI. Troponin neg x2.",
        "Chest pain, atypical, likely MSK.",
        "CP radiating to L arm, EKG ordered.",
    ],
    "respiratory_infection": [
        "URI sx x3 days, cough, low grade fever.",
        "Influenza A positive, started oseltamivir.",
        "pt c/o sore throat & congestion, likely viral. No asthma hx.",
    ],
    "back_pain": [
        "LBP after lifting, no radiculopathy.",
        "Low back pain, chronic, PT referral.",
        "pt c/o back pain. Denies numbness.",
    ],
}


# ---------------------------------------------------------------------------
# Step 1: true people (ground truth)
# ---------------------------------------------------------------------------

@dataclass
class Person:
    """One real (fake!) human. Only the answer key ever sees person_id."""
    person_id: str
    first_name: str
    last_name: str
    dob: date
    sex: str          # "F" or "M" in the ground truth; raw data spells it many ways
    phone_digits: str # 10 digits, no formatting
    zip_code: str
    insurance: str    # canonical payer name


def random_date(rng, start, end):
    """Uniform random date between start and end (inclusive)."""
    return start + timedelta(days=rng.randint(0, (end - start).days))


def random_phone(rng):
    # 555-0100 to 555-0199 is reserved for fiction, so these can't be real numbers.
    # Returns 10 digits, e.g. "6105550142".
    return rng.choice(AREA_CODES) + "55501" + f"{rng.randint(0, 99):02d}"


def shift_years(d, years):
    """Move a date by N years. Feb 29 becomes Feb 28 if the target year isn't a leap year."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(year=d.year + years, day=28)


def make_people(rng, n_people, counts):
    """Create n_people true persons, including twins and household members."""
    people = []
    for i in range(n_people):
        sex = rng.choice(["F", "M"])
        people.append(Person(
            person_id=f"P{i + 1:05d}",
            first_name=rng.choice(FEMALE_NAMES if sex == "F" else MALE_NAMES),
            last_name=rng.choice(LAST_NAMES),
            dob=random_date(rng, date(1930, 1, 1), date(2024, 12, 31)),
            sex=sex,
            phone_digits=random_phone(rng),
            zip_code=rng.choice(ZIP_CODES),
            insurance=rng.choice(list(INSURANCE_VARIANTS)),
        ))

    # Pick people for the traps from a shuffled copy, in separate slices,
    # so nobody is in two traps at once and the counts stay exact.
    shuffled = people[:]
    rng.shuffle(shuffled)
    n_twins = max(1, int(n_people * RATE_TWIN_PAIRS))
    n_household = int(n_people * RATE_HOUSEHOLD)
    twin_slice = shuffled[: 2 * n_twins]
    household_slice = shuffled[2 * n_twins: 2 * n_twins + 2 * n_household]

    # TRAP 1: twins. Same last name, DOB, phone and ZIP, different first name.
    # A matcher that ignores first names will wrongly merge two real people.
    for twin_a, twin_b in zip(twin_slice[0::2], twin_slice[1::2]):
        twin_b.last_name, twin_b.dob = twin_a.last_name, twin_a.dob
        twin_b.phone_digits, twin_b.zip_code = twin_a.phone_digits, twin_a.zip_code
        if twin_b.first_name == twin_a.first_name:  # twins don't share a first name
            pool = FEMALE_NAMES if twin_b.sex == "F" else MALE_NAMES
            twin_b.first_name = rng.choice([n for n in pool if n != twin_a.first_name])
        counts["twin_pairs"] += 1

    # TRAP 2: households. Same last name, phone and ZIP; different DOB and first name.
    for member, relative in zip(household_slice[0::2], household_slice[1::2]):
        member.last_name = relative.last_name
        member.phone_digits, member.zip_code = relative.phone_digits, relative.zip_code
        counts["household_members_sharing_phone"] += 1

    return people


# ---------------------------------------------------------------------------
# Small "make it messy" helpers. Each one does ONE kind of damage.
# ---------------------------------------------------------------------------

def maybe_missing(rng, value, rate, counts, field):
    """Replace a value with one of the six 'missing' spellings."""
    if rng.random() < rate:
        counts[f"missing_{field}"] += 1
        return rng.choice(MISSING_TOKENS)
    return value


def typo(rng, name):
    """Swap two neighbouring letters: 'Smith' -> 'Smtih'."""
    if len(name) < 4:
        return name
    i = rng.randint(1, len(name) - 3)
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


def format_date(rng, d, style):
    """Render a date in one of several real-world formats."""
    if style == "iso":
        return d.strftime("%Y-%m-%d")              # 1985-03-14
    if style == "us":
        return f"{d.month}/{d.day}/{d.year}"       # 3/14/1985
    if style == "us_2digit":
        return f"{d.month}/{d.day}/{d.strftime('%y')}"  # 3/14/85  <- ambiguous century!
    if style == "dd_mon_yyyy":
        return d.strftime("%d-%b-%Y")              # 14-Mar-1985
    if style == "long":
        return d.strftime("%B %d, %Y")             # March 14, 1985
    raise ValueError(style)


# Each system has its own date "habit", plus some random human error.
DATE_STYLE_WEIGHTS = {
    "clinic_a":  {"iso": 0.80, "us": 0.10, "dd_mon_yyyy": 0.05, "long": 0.05},
    "clinic_b":  {"us": 0.70, "us_2digit": 0.20, "iso": 0.10},
    "er_system": {"iso": 0.25, "us": 0.25, "us_2digit": 0.15, "dd_mon_yyyy": 0.20, "long": 0.15},
}


def pick_weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def format_phone(rng, digits):
    a, b, c = digits[:3], digits[3:6], digits[6:]
    return rng.choice([f"({a}) {b}-{c}", f"{a}-{b}-{c}", digits, f"+1 {a}.{b}.{c}"])


def format_sex(rng, sex):
    return rng.choice({"F": ["F", "f", "Female", "FEMALE"],
                       "M": ["M", "m", "Male", "MALE"]}[sex])


def format_name(system, name):
    """clinic_b stores names in UPPERCASE; the others use Title case."""
    return name.upper() if system == "clinic_b" else name


def make_mrn(system, rng, used):
    """Each system has its own MRN format. MRNs are unique within a system."""
    while True:
        if system == "clinic_a":
            mrn = f"A{rng.randint(0, 999999):06d}"
        elif system == "clinic_b":
            mrn = f"{rng.randint(0, 9999999):08d}"   # 8 digits, often leading zeros
        else:
            mrn = f"ER-{rng.randint(0, 99999):05d}"
        if mrn not in used:
            used.add(mrn)
            return mrn


# ---------------------------------------------------------------------------
# Step 2: registrations (the raw patients table)
# ---------------------------------------------------------------------------

def make_patient_row(rng, person, system, mrn, record_id, is_variant, counts):
    """
    Turn a true Person into one raw registration row.
    is_variant=True means this is a 2nd/3rd registration of the same person,
    so it gets extra human-error damage (nicknames, typos).
    """
    first, last, dob = person.first_name, person.last_name, person.dob

    if is_variant:
        if first in NICKNAMES and rng.random() < 0.35:
            first = NICKNAMES[first]
            counts["nickname_used"] += 1
        if rng.random() < 0.15:
            last = typo(rng, last)
            counts["last_name_typo"] += 1

    # Impossible DOBs: a "1985" typed as "2085", or "1980" as "1880".
    if rng.random() < RATE_BAD_DOB:
        dob = shift_years(dob, 100 if rng.random() < 0.5 else -100)
        counts["impossible_dob"] += 1

    dob_style = pick_weighted(rng, DATE_STYLE_WEIGHTS[system])
    if dob_style == "us_2digit":
        counts["two_digit_year_dob"] += 1

    zip_code = person.zip_code + (f"-{rng.randint(1000, 9999)}" if rng.random() < 0.15 else "")

    return {
        "source_system": system,
        "source_record_id": record_id,
        "mrn": mrn,
        "first_name": maybe_missing(rng, format_name(system, first), 0.01, counts, "first_name"),
        "last_name": format_name(system, last),
        "dob": maybe_missing(rng, format_date(rng, dob, dob_style), 0.02, counts, "dob"),
        "sex": maybe_missing(rng, format_sex(rng, person.sex), RATE_MISSING, counts, "sex"),
        "phone": maybe_missing(rng, format_phone(rng, person.phone_digits), RATE_MISSING, counts, "phone"),
        "zip": maybe_missing(rng, zip_code, RATE_MISSING, counts, "zip"),
        "insurance": maybe_missing(rng, rng.choice(INSURANCE_VARIANTS[person.insurance]),
                                   RATE_MISSING, counts, "insurance"),
        # System-generated timestamp: machines write these, so they are clean.
        "record_updated_at": datetime.combine(
            random_date(rng, date(2022, 1, 1), EXTRACT_DATE),
            datetime.min.time()).replace(hour=rng.randint(0, 23), minute=rng.randint(0, 59)
        ).strftime("%Y-%m-%dT%H:%M:%S"),
    }


def make_registrations(rng, people, counts):
    """
    Register each person in 1-3 systems. Returns:
      patient_rows - the raw patients table
      identity     - answer key rows: source_record_id -> person_id
      regs         - (person, system, mrn) for building encounters
    """
    patient_rows, identity, regs = [], [], []
    used_mrns = set()
    next_record = 1

    for person in people:
        n_systems = pick_weighted(rng, SYSTEM_COUNT_WEIGHTS)
        systems = rng.sample(SOURCE_SYSTEMS, n_systems)
        if n_systems > 1:
            counts["people_in_multiple_systems"] += 1

        # Same-system re-registration: the front desk didn't find the old chart
        # and created a new one with a NEW MRN. Very common in real life.
        if rng.random() < RATE_SAME_SYSTEM_REREGISTER:
            systems.append(systems[0])
            counts["same_system_new_mrn"] += 1

        for i, system in enumerate(systems):
            mrn = make_mrn(system, rng, used_mrns)
            record_id = f"REC-{next_record:06d}"
            next_record += 1
            patient_rows.append(make_patient_row(rng, person, system, mrn, record_id,
                                                 is_variant=(i > 0), counts=counts))
            kind = "first" if i == 0 else ("same_system_new_mrn" if system in systems[:i] else "other_system")
            identity.append({"source_record_id": record_id, "person_id": person.person_id,
                             "registration_kind": kind})
            regs.append((person, system, mrn))

    return patient_rows, identity, regs


# ---------------------------------------------------------------------------
# Step 3: encounters (visits)
# ---------------------------------------------------------------------------

def make_vitals(rng, category, counts):
    """Plausible vitals, sometimes missing, sometimes impossible (999, 0)."""
    systolic = int(rng.gauss(150 if category == "hypertension" else 125, 15))
    diastolic = int(rng.gauss(92 if category == "hypertension" else 79, 9))
    heart_rate = int(rng.gauss(78, 12))
    if rng.random() < RATE_BAD_VITALS:
        systolic = rng.choice([999, 0, -1, 1200])
        counts["impossible_vitals"] += 1
    vitals = {"systolic_bp": str(systolic), "diastolic_bp": str(diastolic), "heart_rate": str(heart_rate)}
    if rng.random() < 0.10:  # vitals not taken at all
        counts["missing_vitals"] += 1
        vitals = {k: rng.choice(MISSING_TOKENS) for k in vitals}
    return vitals


def make_note(rng, category, person, mrn, counts):
    """Pick a note template; sometimes leak PHI into it, sometimes leave it empty."""
    if rng.random() < RATE_EMPTY_NOTE:
        counts["empty_note"] += 1
        return rng.choice(MISSING_TOKENS)
    note = rng.choice(NOTE_TEMPLATES[category])
    if rng.random() < RATE_PHI_IN_NOTE:
        counts["phi_leaked_into_note"] += 1
        d = person.phone_digits
        note += rng.choice([
            f" Pt reachable at {d[:3]}-{d[3:6]}-{d[6:]}.",
            f" Spoke with pt's spouse {rng.choice(FEMALE_NAMES + MALE_NAMES)} {person.last_name}.",
            f" See prior chart MRN {mrn}.",
        ])
    return note


def make_encounter_row(rng, encounter_id, person, system, mrn, counts):
    category = rng.choice(list(NOTE_TEMPLATES))
    if system == "er_system":
        enc_type = rng.choices(["emergency", "inpatient"], weights=[0.8, 0.2])[0]
    else:
        enc_type = "outpatient"

    earliest = max(date(2022, 1, 1), person.dob)  # can't visit before being born
    admit = random_date(rng, earliest, EXTRACT_DATE)
    stay_days = rng.randint(1, 10) if enc_type == "inpatient" else 0
    discharge = min(admit + timedelta(days=stay_days), EXTRACT_DATE)

    if rng.random() < RATE_DISCHARGE_BEFORE_ADMIT:  # dates swapped at data entry
        admit, discharge = admit + timedelta(days=rng.randint(1, 5)), admit
        counts["discharge_before_admit"] += 1

    style = pick_weighted(rng, DATE_STYLE_WEIGHTS[system])

    # Excel-style damage: clinic_b MRN "00412345" exported as "412345".
    mrn_as_written = mrn
    if system == "clinic_b" and mrn.startswith("0") and rng.random() < RATE_MRN_LEADING_ZEROS_LOST:
        mrn_as_written = mrn.lstrip("0")
        counts["mrn_leading_zeros_lost"] += 1

    row = {
        "encounter_id": encounter_id,
        "source_system": system,
        "patient_mrn": mrn_as_written,
        "encounter_type": enc_type,
        "admit_date": format_date(rng, admit, style),
        "discharge_date": format_date(rng, discharge, style),
        "diagnosis_note": make_note(rng, category, person, mrn, counts),
        **make_vitals(rng, category, counts),
    }
    return row, category


def make_encounters(rng, regs, counts):
    rows, labels = [], []
    next_id = 1
    for person, system, mrn in regs:
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 4])):
            enc_id = f"ENC-{next_id:06d}"
            next_id += 1
            row, category = make_encounter_row(rng, enc_id, person, system, mrn, counts)
            rows.append(row)
            labels.append({"encounter_id": enc_id, "person_id": person.person_id,
                           "true_category": category})

    # Orphans: encounters whose MRN matches no registration (lost or mistyped upstream).
    n_orphans = int(len(rows) * RATE_ORPHAN_ENCOUNTER)
    used_mrns = set(mrn for _, _, mrn in regs)  # so the bogus MRN never hits a real one
    for _ in range(n_orphans):
        fake_person, system, _ = rng.choice(regs)
        bogus_mrn = make_mrn(system, rng, used_mrns)
        enc_id = f"ENC-{next_id:06d}"
        next_id += 1
        row, category = make_encounter_row(rng, enc_id, fake_person, system, bogus_mrn, counts)
        rows.append(row)
        labels.append({"encounter_id": enc_id, "person_id": "", "true_category": category})
        counts["orphan_encounters"] += 1

    return rows, labels


# ---------------------------------------------------------------------------
# Step 4: final damage + writing files
# ---------------------------------------------------------------------------

def add_exact_duplicates(rng, rows, counts, name):
    """Simulate a batch job that loaded some rows twice, then shuffle the file."""
    dupes = [dict(r) for r in rows if rng.random() < RATE_EXACT_DUP_ROW]
    counts[f"exact_duplicate_{name}_rows"] = len(dupes)
    out = rows + dupes
    rng.shuffle(out)  # real exports aren't sorted nicely
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate messy synthetic healthcare data.")
    parser.add_argument("--seed", type=int, default=42, help="same seed = same files")
    parser.add_argument("--people", type=int, default=500, help="number of true people")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output folder")
    args = parser.parse_args()

    # One random generator, seeded once. Every random choice flows from it,
    # which is why the output is exactly reproducible.
    rng = random.Random(args.seed)
    counts = Counter()

    people = make_people(rng, args.people, counts)
    patient_rows, identity, regs = make_registrations(rng, people, counts)
    encounter_rows, labels = make_encounters(rng, regs, counts)

    patient_rows = add_exact_duplicates(rng, patient_rows, counts, "patient")
    encounter_rows = add_exact_duplicates(rng, encounter_rows, counts, "encounter")

    raw, key = args.out / "raw", args.out / "answer_key"
    write_csv(raw / "patients.csv", patient_rows)
    write_csv(raw / "encounters.csv", encounter_rows)
    write_csv(key / "patient_identity.csv", identity)
    write_csv(key / "encounter_labels.csv", labels)

    manifest = {
        "seed": args.seed,
        "extract_date": EXTRACT_DATE.isoformat(),
        "true_people": len(people),
        "raw_patient_rows": len(patient_rows),
        "raw_encounter_rows": len(encounter_rows),
        "mess_injected": dict(sorted(counts.items())),
    }
    (key / "mess_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"True people:        {len(people)}")
    print(f"Raw patient rows:   {len(patient_rows)}  -> {raw / 'patients.csv'}")
    print(f"Raw encounter rows: {len(encounter_rows)}  -> {raw / 'encounters.csv'}")
    print("Mess injected:")
    for k, v in sorted(counts.items()):
        print(f"  {k:<36} {v}")


if __name__ == "__main__":
    main()
