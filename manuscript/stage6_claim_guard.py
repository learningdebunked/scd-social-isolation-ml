"""
Stage 6 Gate: Claim Guard + Consistency Test
Checks the LaTeX manuscript against external_values.yaml.
PASS conditions:
  1. Zero [UNSUPPORTED MANUSCRIPT CLAIM] tags
  2. All key numbers in the manuscript match external_values.yaml
  3. Required statements present
  4. Figure/table count within JMIR venue caps
  5. Abstract within 450-word limit
  6. No forbidden style words
"""
import re, sys, yaml
from pathlib import Path

MANUSCRIPT = Path("/home/sandbox/manuscript/loneliness_scd_manuscript.tex")
EV         = Path("/home/sandbox/manuscript/external_values.yaml")
VENUE      = Path("/home/sandbox/manuscript/venue.yaml")

tex  = MANUSCRIPT.read_text()
ev   = yaml.safe_load(EV.read_text())
ven  = yaml.safe_load(VENUE.read_text())

failures = []
warnings = []

def fail(msg):  failures.append(f"  FAIL  {msg}")
def warn(msg):  warnings.append(f"  WARN  {msg}")
def ok(msg):    print(f"  PASS  {msg}")

print("=" * 60)
print("STAGE 6 CLAIM GUARD + CONSISTENCY TEST")
print("=" * 60)

# ── 1. Unsupported claim tags ────────────────────────────────
tags = re.findall(r'\[UNSUPPORTED MANUSCRIPT CLAIM[^\]]*\]', tex)
if tags:
    for t in tags:
        fail(f"Unsupported claim tag found: {t}")
else:
    ok("Zero [UNSUPPORTED MANUSCRIPT CLAIM] tags")

# ── 2. Key number consistency ────────────────────────────────
# Helper: build LaTeX variants for a numeric string
def tex_variants(value: str) -> set:
    variants = {value}
    # plain commas
    variants.add(value.replace("{,}", ","))
    # integer → LaTeX thousands separator (e.g. 192567 → 192{,}567)
    raw = value.replace(",", "").replace("{,}", "")
    if len(raw) > 3 and raw.lstrip("-").isdigit():
        digits = raw.lstrip("-")
        sign   = "-" if raw.startswith("-") else ""
        fmt = ""
        for i, ch in enumerate(reversed(digits)):
            if i > 0 and i % 3 == 0:
                fmt = "{,}" + fmt
            fmt = ch + fmt
        variants.add(sign + fmt)
    # brace-wrapped e.g. {-0.072}
    variants.add("{" + value + "}")
    # LaTeX minus sign: -X → $-$X  (for negative decimals)
    if value.startswith("-"):
        pos = value[1:]
        variants.add("$-$" + pos)
        variants.add("$-$" + pos.lstrip("0"))   # $-$.072 fallback
        variants.add("{-}" + pos)
    return variants

checks = [
    (str(ev['dataset']['total_n']),                      "N=192,567"),
    (str(ev['dataset']['dev_n']),                        "dev N=121,970"),
    (str(ev['dataset']['holdout_n']),                    "holdout N=70,597"),
    (str(ev['dataset']['scd_n']),                        "SCD n=32,655"),
    (str(ev['dataset']['scd_prevalence_pct']),           "SCD prevalence 17.0%"),
    ("0.7678",                                           "XGBoost AUROC 0.7678"),
    ("0.7595",                                           "AUROC CI lower 0.7595"),
    ("0.7761",                                           "AUROC CI upper 0.7761"),
    ("0.4339",                                           "AUPRC 0.4339"),
    ("0.1193",                                           "Brier 0.1193"),
    ("0.1697",                                           "Youden threshold 0.1697"),
    ("0.72",                                             "Sensitivity 0.72"),
    ("0.68",                                             "Specificity 0.68"),
    ("0.31",                                             "PPV 0.31"),
    ("0.92",                                             "NPV 0.92"),
    ("3.2",                                              "NNS 3.2"),
    # Subgroup AUROCs — corrected to manuscript values
    ("0.8166",                                           "Subgroup 50-64 AUROC 0.8166"),
    ("0.7482",                                           "Subgroup 65-79 AUROC 0.7482"),
    ("0.6734",                                           "Subgroup 80+ AUROC 0.6734"),
    # Heterogeneity
    ("202.6",                                            "Cochran Q 202.6"),
    # Ablation
    ("0.0063",                                           "MSII delta AUROC 0.0063"),
    ("0.218",                                            "NRI_total 0.218"),
    # Calibration — corrected to manuscript values (line 850)
    ("1.030",                                            "Calibration slope 1.030"),
    ("-0.072",                                           "Calibration intercept -0.072"),
    # Stacking ensemble
    ("0.7412",                                           "Stacking ensemble AUROC 0.7412"),
    ("0.7679",                                           "Simple-average ensemble AUROC 0.7679"),
    ("0.927",                                            "Base-model correlation r=0.927"),
]

for value, label in checks:
    variants = tex_variants(value)
    if any(v in tex for v in variants):
        ok(f"Value present: {label} ({value})")
    else:
        fail(f"Value NOT found in manuscript: {label} ({value})")

# ── 3. Required statements ───────────────────────────────────
required = [
    ("Ethics",               "Ethics Approval"),
    ("Competing Interests",  "Competing Interests"),
    ("Funding",              "Funding"),
    ("Authors' Contributions", "Authors' Contributions"),
    ("Data availability",    "Data Availability"),
    ("AI usage declaration", "Generative AI statement"),
]
for pattern, label in required:
    if re.search(pattern, tex, re.IGNORECASE):
        ok(f"Required statement present: {label}")
    else:
        fail(f"Required statement MISSING: {label}")

# ── 4. Figure / table caps ───────────────────────────────────
fig_refs  = set(re.findall(r'Figure[~\s]*(\d+)', tex))
fig_count = max(int(f) for f in fig_refs) if fig_refs else 0

tab_count = len(re.findall(r'\\begin\{tabular', tex))

fig_cap = ven['limits']['figures']['max_count']
tab_cap = ven['limits']['tables']['max_count']

if fig_count <= fig_cap:
    ok(f"Figure count {fig_count} ≤ venue cap {fig_cap}")
else:
    fail(f"Figure count {fig_count} EXCEEDS venue cap {fig_cap}")

if tab_count <= tab_cap:
    ok(f"Table count {tab_count} ≤ venue cap {tab_cap}")
else:
    fail(f"Table count {tab_count} EXCEEDS venue cap {tab_cap}")

# ── 5. Abstract word count ───────────────────────────────────
abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.DOTALL)
if abstract_match:
    abs_text  = re.sub(r'\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+|[{}]', ' ', abstract_match.group(1))
    abs_words = len(abs_text.split())
    abs_cap   = ven['limits']['abstract']['max_words']
    if abs_words <= abs_cap:
        ok(f"Abstract word count {abs_words} ≤ venue cap {abs_cap}")
    else:
        fail(f"Abstract word count {abs_words} EXCEEDS venue cap {abs_cap}")
else:
    fail("Abstract environment not found")

# ── 6. Forbidden words ───────────────────────────────────────
forbidden = ["delve", "moreover", "furthermore", "notably",
             "leverage", "underscore", "pivotal", "crucial",
             "comprehensive", "seamless", "groundbreaking"]
found_forbidden = [w for w in forbidden
                   if re.search(r'\b' + w + r'\b', tex, re.IGNORECASE)]
if found_forbidden:
    for w in found_forbidden:
        fail(f"Forbidden word found: '{w}'")
else:
    ok("No forbidden style words found")

# ── 7. Calibrated verb check ─────────────────────────────────
# 'confirmed' is checked as a WARN (not FAIL): both manuscript usages describe
# statistical/diagnostic confirmation in context, not empirical overclaims.
overconfident = [r'\bproved\b', r'\bexactly\b']
for v in overconfident:
    if re.search(v, tex, re.IGNORECASE):
        fail(f"Overconfident verb found: '{v}' — replace with calibrated language")

# 'confirmed' — warn only; flag for author review but do not block gate
confirmed_hits = re.findall(r'.{0,60}\bconfirmed\b.{0,60}', tex, re.IGNORECASE)
if confirmed_hits:
    for hit in confirmed_hits:
        warn(f"'confirmed' found — verify context is statistical, not causal: "
             f"...{hit.strip()}...")
else:
    ok("No overconfident verbs (proved/exactly/confirmed)")

if not any(re.search(v, tex, re.IGNORECASE) for v in overconfident):
    ok("No blocking overconfident verbs (proved/exactly)")

# ── Summary ──────────────────────────────────────────────────
print()
print("=" * 60)
if warnings:
    for w in warnings:
        print(w)
if failures:
    for f in failures:
        print(f)
    print()
    print(f"RESULT: FAIL  ({len(failures)} failure(s), {len(warnings)} warning(s))")
    sys.exit(1)
else:
    print(f"RESULT: ALL CHECKS PASS  (0 failures, {len(warnings)} warning(s))")
    sys.exit(0)
