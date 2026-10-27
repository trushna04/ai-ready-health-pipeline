"""
match_patients.py  (Phase 1)

Decides which patient records belong to the same real person.

Three steps:
  1. BLOCKING: only compare records that share something (same DOB, same phone,
     or same first name + last initial). Comparing every record to every other
     is fine for 700 rows but impossible for 70 million.
  2. SCORING: compare each candidate pair field by field. Some disagreements
     are VETOES (two different first names = two different people, even when
     DOB, phone and address all match: that's how twins look).
  3. DECIDING: each pair becomes auto_match, review, or non_match.
     Only auto_match pairs are merged. Review pairs go to a human queue;
     a wrong merge mixes two people's medical histories, which is worse
     than leaving a duplicate.

Input records are cleaned dicts (see pipeline.py). Nothing here reads files.
"""

from collections import defaultdict
from itertools import combinations

# Points each agreeing field adds. DOB and first name carry the most weight
# because they are the fields that tell twins and family members apart.
WEIGHTS = {
    "first_name": 3,
    "last_name_exact": 2, "last_name_typo": 1,
    "dob_exact": 4, "dob_same_month_day": 1,
    "phone": 2,
    "zip": 1,
    "sex": 1,
}
REVIEW_THRESHOLD = 6


def edit_distance_at_most_one(a, b):
    """
    True if a and b differ by at most one typo: a changed, added or removed
    letter, or two neighbouring letters swapped ('Smith' vs 'Smtih').
    """
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 1:
            return True
        return (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                and a[diffs[0]] == b[diffs[1]] and a[diffs[1]] == b[diffs[0]])
    shorter, longer = sorted((a, b), key=len)
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def blocking_keys(rec):
    """The 'buckets' a record is placed in. Two records are compared only if they share one."""
    keys = []
    if rec["dob"]:
        keys.append(("dob", rec["dob"]))
    if rec["phone"]:
        keys.append(("phone", rec["phone"]))
    if rec["formal_first_name"] and rec["last_name"]:
        keys.append(("name", rec["formal_first_name"], rec["last_name"][0]))
    return keys


def candidate_pairs(records):
    blocks = defaultdict(list)
    for i, rec in enumerate(records):
        for key in blocking_keys(rec):
            blocks[key].append(i)
    pairs = set()
    for members in blocks.values():
        pairs.update(combinations(sorted(members), 2))
    return sorted(pairs)


def compare(a, b):
    """
    Compare two records. Returns (decision, score, reasons).
    reasons is a short human-readable list, written for the review queue.
    """
    score, reasons = 0, []

    def both(field):
        return a[field] is not None and b[field] is not None

    # First name (nicknames already converted to the formal name).
    first_agrees = both("formal_first_name") and a["formal_first_name"] == b["formal_first_name"]
    if both("formal_first_name") and not first_agrees:
        return "non_match", 0, ["first names differ"]
    if first_agrees:
        score += WEIGHTS["first_name"]
        reasons.append("first name agrees" + (" (via nickname)" if a["first_name"] != b["first_name"] else ""))
    else:
        reasons.append("first name missing")

    # Last name: exact, or one typo away.
    if a["last_name"] == b["last_name"]:
        score += WEIGHTS["last_name_exact"]
    elif edit_distance_at_most_one(a["last_name"], b["last_name"]):
        score += WEIGHTS["last_name_typo"]
        reasons.append(f"last name one typo apart ({a['last_name']} / {b['last_name']})")
    else:
        return "non_match", 0, ["last names differ"]

    # DOB: exact is strong. Same month and day but a different year is a likely
    # year typo, so it's weak evidence, not a veto. Anything else is a veto.
    dob_exact = both("dob") and a["dob"] == b["dob"]
    if dob_exact:
        score += WEIGHTS["dob_exact"]
    elif both("dob") and (a["dob"].month, a["dob"].day) == (b["dob"].month, b["dob"].day):
        score += WEIGHTS["dob_same_month_day"]
        reasons.append("DOB same month/day, different year")
    elif both("dob"):
        return "non_match", 0, ["DOBs differ"]
    else:
        reasons.append("DOB missing")

    if both("sex") and a["sex"] != b["sex"]:
        return "non_match", 0, ["sex differs"]
    if both("sex"):
        score += WEIGHTS["sex"]

    # Phone and ZIP only ADD confidence. Different phone is not a veto: people change numbers.
    # Matching phone alone proves nothing: households and twins share one.
    if both("phone") and a["phone"] == b["phone"]:
        score += WEIGHTS["phone"]
        reasons.append("phone agrees")
    if both("zip") and a["zip"] == b["zip"]:
        score += WEIGHTS["zip"]

    # Auto-merge needs BOTH first name and exact DOB. Without them we can't
    # rule out a twin or a same-named relative, however high the score.
    if first_agrees and dob_exact:
        return "auto_match", score, reasons
    if score >= REVIEW_THRESHOLD:
        return "review", score, reasons
    return "non_match", score, reasons


class UnionFind:
    """Groups records into clusters: if A=B and B=C then A, B, C are one person."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        self.parent[self.find(j)] = self.find(i)


def match(records):
    """
    Returns:
      golden_ids   - list, golden_patient_id for each record (same order as input)
      pair_results - list of dicts, one per compared pair (for review queue + audit)
    """
    uf = UnionFind(len(records))
    pair_results = []
    for i, j in candidate_pairs(records):
        decision, score, reasons = compare(records[i], records[j])
        if decision == "auto_match":
            uf.union(i, j)
        pair_results.append({"i": i, "j": j, "decision": decision, "score": score,
                             "reasons": "; ".join(reasons)})

    # Name each cluster after its earliest record, so IDs are stable between runs.
    clusters = defaultdict(list)
    for i in range(len(records)):
        clusters[uf.find(i)].append(i)
    golden_ids = [None] * len(records)
    for members in clusters.values():
        first_rec = min(records[m]["source_record_id"] for m in members)
        for m in members:
            golden_ids[m] = "GP-" + first_rec.split("-")[1]
    return golden_ids, pair_results
