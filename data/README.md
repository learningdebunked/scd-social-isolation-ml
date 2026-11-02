# Data Directory

Raw BRFSS data files are **not included** in this repository due to their size
(~450–500 MB per year). They must be downloaded from the CDC before running the pipeline.

---

## Download Instructions

### BRFSS 2023

1. Visit: https://www.cdc.gov/brfss/annual_data/annual_2023.html
2. Under the "Data" section, download **2023 BRFSS Data (XPT)**
3. Save as: `data/raw/LLCP2023.XPT`
4. Also download the **2023 BRFSS Codebook** (PDF) for variable reference

Direct link (may change — verify on the CDC page if broken):
```
https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip
```

### BRFSS 2024

1. Visit: https://www.cdc.gov/brfss/annual_data/annual_2024.html
2. Under the "Data" section, download **2024 BRFSS Data (XPT)**
3. Save as: `data/raw/LLCP2024.XPT`

Direct link (verify on CDC page):
```
https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip
```

### Automated download

A convenience shell script is provided:

```bash
bash data/download_brfss.sh
```

This will download and unzip both years into `data/raw/`.

---

## Key Variables Used

### Outcome variable

| Variable | Question | Coding used |
|----------|----------|-------------|
| `CIMEMLOS` | "During the past 12 months, have you experienced confusion or memory loss that is happening more often or is getting worse?" | 1=Yes (SCD=1); 2=No (SCD=0); 7,9,blank=Excluded |

> **Note:** `CIMEMLOS` is part of the optional **Cognitive Decline Module** of BRFSS.
> It is fielded in all 50 states + DC in 2023 and 2024, but response rates vary by state.

### Survey design variables

| Variable | Description |
|----------|-------------|
| `_LLCPWT` | Final rake weight for national representativeness |
| `_STSTR`  | Stratification variable for complex survey design |
| `_PSU`    | Primary sampling unit |

### Primary social isolation features

| Variable | Question |
|----------|----------|
| `LSATISFY` | Satisfaction with life in general |
| `EMTSUPRT` | Frequency of receiving emotional support |
| `SDHISOLT` | How often felt socially isolated |
| `SDHBOND`  | Sense of belonging in community |

---

## Expected file checksums (after unzip)

```
LLCP2023.XPT   SHA-256: [to be confirmed upon dataset access]
LLCP2024.XPT   SHA-256: [to be confirmed upon dataset access]
```

---

## Processed data (auto-generated)

Running the pipeline creates the following in `data/processed/`:

```
data/processed/
├── brfss2023_clean.parquet        # Cleaned 2023 data (aged 50+, SCD ascertained)
├── brfss2024_clean.parquet        # Cleaned 2024 data
├── brfss2023_features.parquet     # Feature-engineered 2023 dataset
└── brfss2024_features.parquet     # Feature-engineered 2024 dataset
```

These files are also excluded from the repository (`.gitignore`) to avoid
re-distributing derived CDC data. They can be fully reconstructed from
the raw XPT files using the provided preprocessing scripts.
