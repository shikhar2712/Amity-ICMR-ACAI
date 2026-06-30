"""
DR Pathogen List
================
Canonical list of pathogens for the "Doctor Recommendation & Laboratory
Variables" section — used by the **Suspected Pathogens** and **Confirmed
Pathogen** dropdowns (on both the Prediction page and the View Records →
Update DR form).

Source: ICMR-provided ``DR_Pathogen_List.csv`` (100 pathogens).

This is FRONTEND reference data only. It is intentionally INDEPENDENT of the
prediction model's output classes (``model_handler.VIRUS_MAPPING`` /
``OTHER_VIRUS_MAPPING``): a clinician may suspect or confirm a pathogen the
model was never trained to predict. Nothing in this module touches the model,
its ``.pth`` weights, or the prediction pipeline.

To update the list, edit ``DR_Pathogen_List.csv`` (a ``Pathogen Name`` column)
and redeploy. If the CSV is missing or unreadable on the host, the embedded
fallback below is used so the dropdowns never come up empty.
"""
from pathlib import Path
import csv

# Path to the source CSV (kept alongside the other reference CSVs at repo root).
_CSV_PATH = Path(__file__).resolve().parent / "DR_Pathogen_List.csv"

# Embedded fallback — kept in sync with DR_Pathogen_List.csv. Order matches the
# CSV (ICMR's curated ordering) and is preserved as-is in the dropdowns.
EMBEDDED_PATHOGENS = [
    "Acanthamoeba sp.",
    "Acinetobacter baumannii",
    "Adenovirus",
    "Astrovirus",
    "Bacillus anthracis",
    "Balamuthia mandrillaris",
    "BK virus",
    "Bordetella parapertussis",
    "Bordetella pertussis",
    "Campylobacter Spp",
    "Chandipura virus",
    "Chikungunya virus",
    "Corynebacterium diphtheria",
    "Coxsackie virus",
    "Crimean-Congo haemorrhagic fever (CCHF)",
    "Cytomegalovirus (CMV)",
    "Dengue virus",
    "Dengue virus - 1",
    "Dengue virus - 2",
    "Dengue virus - 3",
    "Dengue virus - 4",
    "Ebola virus",
    "Enterovirus",
    "Epstein-Barr virus",
    "Hantavirus",
    "Hepatitis A virus",
    "Hepatitis B virus",
    "Hepatitis C virus",
    "Hepatitis D virus",
    "Hepatitis E virus",
    "Herpes simplex virus (HSV)",
    "Herpes simplex virus-1 (HSV-1)",
    "Herpes simplex virus-2 (HSV-2)",
    "Human bocavirus",
    "Human Coronavirus 229E (HCoV-229E)",
    "Human Coronavirus HKU1 (HCoV-HKU1)",
    "Human Coronavirus NL63 (HCoV-NL63)",
    "Human Coronavirus OC43 (HCoV-OC43)",
    "Human Herpesvirus 6 (HHV-6)",
    "Human Herpesvirus 7 (HHV-7)",
    "Human Herpesvirus 8 (HHV-8)",
    "Human metapneumovirus",
    "Human papillomavirus (HPV)",
    "Human parechovirus",
    "Human Parvovirus",
    "Influenza A (H1N1)",
    "Influenza A (H3N2)",
    "Influenza A (H5N1)",
    "Influenza A not subtyped",
    "Influenza B (Victoria)",
    "Influenza B (Yamagata)",
    "Influenza B not subtyped",
    "Influenza Unsubtypable",
    "Japanese encephalitis",
    "Klebsiella pneumoniae",
    "Kyasanur forest disease",
    "Lassa virus",
    "Leptospira",
    "Listeria monocytogenes",
    "Malaria falciparum",
    "Malaria mixed",
    "Malaria vivax",
    "Marburg virus",
    "Measles virus",
    "Mpox virus",
    "Mumps virus",
    "Mycoplasma pneumoniae",
    "Naegleria fowleri",
    "Neisseria meningitidis",
    "Nipah virus",
    "Norovirus",
    "Parainfluenza virus",
    "Parainfluenza virus - 1",
    "Parainfluenza virus - 2",
    "Parainfluenza virus - 3",
    "Parainfluenza virus - 4",
    "Rabies virus",
    "Reovirus",
    "Respiratory Syncytial Virus (RSV)",
    "Respiratory Syncytial Virus-A (RSV-A)",
    "Respiratory Syncytial Virus-B (RSV-B)",
    "Rhinovirus",
    "Rickettsia Spp",
    "Rotavirus",
    "Rubella virus",
    "Salmonella paratyphi A",
    "Salmonella paratyphi B",
    "Salmonella paratyphi C",
    "Salmonella typhi",
    "Sapovirus",
    "Sappinia sp.",
    "SARS-Cov-2",
    "Scrub typhus (Orientia tsutsugamushi)",
    "Shigella",
    "Streptococcus pyogenes",
    "Toxoplasmosis",
    "Varicella zoster virus (VZV)",
    "Vibrio cholerae O1",
    "West Nile virus (WNV)",
    "Zika virus",
]


def _load_from_csv(path):
    """Read pathogen names from the CSV's 'Pathogen Name' column (order preserved,
    blanks skipped, duplicates removed). Raises on any read/parse error."""
    names = []
    seen = set()
    # utf-8-sig transparently strips a BOM if the CSV was saved from Excel.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("Pathogen Name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _build_pathogen_list():
    """Prefer the CSV (so ICMR can update it without code changes); fall back to
    the embedded list if the CSV is missing or unreadable."""
    try:
        if _CSV_PATH.exists():
            names = _load_from_csv(_CSV_PATH)
            if names:
                return names
    except Exception:
        # Any I/O or parse problem -> safe embedded fallback.
        pass
    return list(EMBEDDED_PATHOGENS)


# Final, ready-to-use list for the dropdowns. Computed once at import time.
DR_SUSPECTED_PATHOGENS = _build_pathogen_list()
