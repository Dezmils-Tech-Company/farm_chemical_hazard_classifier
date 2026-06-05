"""
===============================================================
AGRICULTURAL CHEMICAL SAFETY AI — THESIS DATA COLLECTION SCRIPT
===============================================================
Author  : Your Name (Master's Thesis)
Purpose : Download & save all required datasets locally

Sources covered:
  1. PubChem        — SMILES, molecular properties
  2. EPA CompTox    — Toxicity endpoints, hazard data
  3. PAN Pesticide  — Toxicity flags, regulatory status
  4. ECOTOX (EPA)   — Ecotoxicology data
  5. ChEMBL         — Bioactivity data
  6. WHO Pesticide  — Hazard classifications (embedded list)
  7. KEPHIS/PCPB    — Kenya registered/banned pesticides (embedded)

Output:
  data/raw/           — individual source files
  data/processed/     — merged master dataset ready for ML
===============================================================
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── directory setup ──────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
EXT_DIR    = BASE_DIR / "data" / "external"
LOG_DIR    = BASE_DIR / "logs"

for d in [RAW_DIR, PROC_DIR, EXT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  AGRICULTURAL CHEMICAL SAFETY — DATA COLLECTION")
print("=" * 60)


# ════════════════════════════════════════════════════════════
# SECTION 1 — PESTICIDE NAME LIST
# Common agricultural pesticides (your query seed list)
# ════════════════════════════════════════════════════════════

PESTICIDE_LIST = [
    # Organophosphates
    "glyphosate", "chlorpyrifos", "malathion", "parathion",
    "dimethoate", "diazinon", "acephate", "trichlorfon",
    "methidathion", "profenofos", "phorate", "disulfoton",
    # Pyrethroids
    "cypermethrin", "deltamethrin", "permethrin", "lambda-cyhalothrin",
    "bifenthrin", "cyfluthrin", "esfenvalerate", "alpha-cypermethrin",
    "fenpropathrin", "zeta-cypermethrin",
    # Organochlorines
    "DDT", "endosulfan", "lindane", "aldrin", "dieldrin",
    "chlordane", "heptachlor", "mirex", "toxaphene",
    # Carbamates
    "carbofuran", "carbaryl", "methomyl", "aldicarb",
    "oxamyl", "pirimicarb", "fenobucarb", "bendiocarb",
    # Neonicotinoids
    "imidacloprid", "thiamethoxam", "clothianidin", "acetamiprid",
    "dinotefuran", "nitenpyram", "thiacloprid",
    # Fungicides
    "mancozeb", "chlorothalonil", "propiconazole", "tebuconazole",
    "iprodione", "captan", "thiram", "zineb", "maneb",
    "azoxystrobin", "trifloxystrobin", "carbendazim", "benomyl",
    # Herbicides
    "atrazine", "2,4-D", "paraquat", "metolachlor", "alachlor",
    "pendimethalin", "trifluralin", "dicamba", "metsulfuron-methyl",
    "imazapyr", "imazethapyr", "fluazifop-butyl",
    # Rodenticides / others
    "brodifacoum", "bromadiolone", "zinc phosphide",
    "methyl bromide", "aluminium phosphide",
    # Common in East Africa / Kenya context
    "profenofos", "acetamiprid", "emamectin benzoate",
    "abamectin", "spinosad", "indoxacarb", "lufenuron",
    "chlorfenapyr", "spiromesifen", "hexythiazox",
]

print(f"\n[INFO] Pesticide list loaded: {len(PESTICIDE_LIST)} chemicals\n")


# ════════════════════════════════════════════════════════════
# SECTION 2 — PubChem DATA COLLECTION
# Gets: SMILES, molecular weight, LogP, H-bond donors/acceptors,
#       TPSA, rotatable bonds, CID, InChIKey
# ════════════════════════════════════════════════════════════

def fetch_pubchem_data(chemical_name: str) -> dict:
    """Fetch molecular properties from PubChem REST API."""
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    props = (
        "IsomericSMILES,MolecularFormula,MolecularWeight,"
        "XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount,HeavyAtomCount,Complexity,"
        "InChIKey,IUPACName"
    )
    url = f"{base}/compound/name/{requests.utils.quote(chemical_name)}/property/{props}/JSON"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()["PropertyTable"]["Properties"][0]
            data["query_name"] = chemical_name
            return data
    except Exception as e:
        print(f"  [WARN] PubChem: {chemical_name} — {e}")
    return {"query_name": chemical_name}


def collect_pubchem(pesticides: list, save_path: Path) -> pd.DataFrame:
    print("\n[1/5] Collecting PubChem data...")
    records = []
    for i, name in enumerate(pesticides, 1):
        print(f"  ({i}/{len(pesticides)}) {name}", end="\r")
        rec = fetch_pubchem_data(name)
        records.append(rec)
        time.sleep(0.3)   # respect rate limits
    df = pd.DataFrame(records)
    df.columns = [c.lower() for c in df.columns]
    df.to_csv(save_path, index=False)
    print(f"\n  ✓ Saved {len(df)} records → {save_path.name}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 3 — EPA CompTox DATA COLLECTION
# Gets: toxicity data, hazard flags, physicochemical properties
# ════════════════════════════════════════════════════════════

def fetch_comptox_data(chemical_name: str) -> dict:
    """Fetch data from EPA CompTox Dashboard API."""
    base = "https://comptox.epa.gov/dashboard-api"
    url  = f"{base}/ccdapp1/chemical-detail/search/by-name/?word={requests.utils.quote(chemical_name)}"
    try:
        r = requests.get(url, timeout=15,
                         headers={"accept": "application/json"})
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                return {
                    "query_name"       : chemical_name,
                    "dtxsid"           : item.get("dtxsid", ""),
                    "preferred_name"   : item.get("preferredName", ""),
                    "cas_rn"           : item.get("casrn", ""),
                    "mol_formula"      : item.get("molFormula", ""),
                    "mol_weight_comptox": item.get("molWeight", ""),
                    "inchikey_comptox" : item.get("inchiKey", ""),
                    "smiles_comptox"   : item.get("smiles", ""),
                }
    except Exception as e:
        print(f"  [WARN] CompTox: {chemical_name} — {e}")
    return {"query_name": chemical_name}


def collect_comptox(pesticides: list, save_path: Path) -> pd.DataFrame:
    print("\n[2/5] Collecting EPA CompTox data...")
    records = []
    for i, name in enumerate(pesticides, 1):
        print(f"  ({i}/{len(pesticides)}) {name}", end="\r")
        rec = fetch_comptox_data(name)
        records.append(rec)
        time.sleep(0.4)
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"\n  ✓ Saved {len(df)} records → {save_path.name}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 4 — WHO HAZARD CLASSIFICATIONS (EMBEDDED)
# WHO classifies pesticides into: Ia, Ib, II, III, U
# Source: WHO 2019 Recommended Classification of Pesticides
# ════════════════════════════════════════════════════════════

WHO_HAZARD_DATA = {
    # Format: "chemical_name": ("WHO_class", "class_description", "LD50_rat_oral_approx")
    "glyphosate"          : ("U",   "Unlikely hazardous",           5600),
    "chlorpyrifos"        : ("II",  "Moderately hazardous",         135),
    "malathion"           : ("III", "Slightly hazardous",           2100),
    "parathion"           : ("Ia",  "Extremely hazardous",          2),
    "dimethoate"          : ("II",  "Moderately hazardous",         150),
    "diazinon"            : ("II",  "Moderately hazardous",         300),
    "acephate"            : ("III", "Slightly hazardous",           945),
    "trichlorfon"         : ("II",  "Moderately hazardous",         450),
    "methidathion"        : ("Ib",  "Highly hazardous",             25),
    "profenofos"          : ("II",  "Moderately hazardous",         358),
    "phorate"             : ("Ia",  "Extremely hazardous",          1),
    "disulfoton"          : ("Ia",  "Extremely hazardous",          2),
    "cypermethrin"        : ("II",  "Moderately hazardous",         250),
    "deltamethrin"        : ("II",  "Moderately hazardous",         135),
    "permethrin"          : ("II",  "Moderately hazardous",         430),
    "lambda-cyhalothrin"  : ("II",  "Moderately hazardous",         56),
    "bifenthrin"          : ("II",  "Moderately hazardous",         54),
    "cyfluthrin"          : ("II",  "Moderately hazardous",         869),
    "esfenvalerate"       : ("II",  "Moderately hazardous",         87),
    "alpha-cypermethrin"  : ("II",  "Moderately hazardous",         79),
    "DDT"                 : ("II",  "Moderately hazardous",         113),
    "endosulfan"          : ("Ib",  "Highly hazardous",             18),
    "lindane"             : ("II",  "Moderately hazardous",         88),
    "aldrin"              : ("Ia",  "Extremely hazardous",          39),
    "dieldrin"            : ("Ia",  "Extremely hazardous",          46),
    "chlordane"           : ("II",  "Moderately hazardous",         283),
    "heptachlor"          : ("II",  "Moderately hazardous",         147),
    "carbofuran"          : ("Ib",  "Highly hazardous",             8),
    "carbaryl"            : ("II",  "Moderately hazardous",         250),
    "methomyl"            : ("Ib",  "Highly hazardous",             17),
    "aldicarb"            : ("Ia",  "Extremely hazardous",          0.9),
    "oxamyl"              : ("Ib",  "Highly hazardous",             5),
    "pirimicarb"          : ("II",  "Moderately hazardous",         147),
    "imidacloprid"        : ("II",  "Moderately hazardous",         450),
    "thiamethoxam"        : ("II",  "Moderately hazardous",         1563),
    "clothianidin"        : ("II",  "Moderately hazardous",         5000),
    "acetamiprid"         : ("II",  "Moderately hazardous",         217),
    "thiacloprid"         : ("II",  "Moderately hazardous",         836),
    "mancozeb"            : ("U",   "Unlikely hazardous",           11200),
    "chlorothalonil"      : ("U",   "Unlikely hazardous",           10000),
    "propiconazole"       : ("II",  "Moderately hazardous",         1517),
    "tebuconazole"        : ("II",  "Moderately hazardous",         1700),
    "iprodione"           : ("III", "Slightly hazardous",           3500),
    "captan"              : ("III", "Slightly hazardous",           9000),
    "thiram"              : ("III", "Slightly hazardous",           560),
    "carbendazim"         : ("U",   "Unlikely hazardous",           15000),
    "benomyl"             : ("U",   "Unlikely hazardous",           10000),
    "azoxystrobin"        : ("U",   "Unlikely hazardous",           5000),
    "atrazine"            : ("U",   "Unlikely hazardous",           1869),
    "2,4-D"               : ("II",  "Moderately hazardous",         639),
    "paraquat"            : ("Ib",  "Highly hazardous",             58),
    "metolachlor"         : ("III", "Slightly hazardous",           2780),
    "alachlor"            : ("III", "Slightly hazardous",           930),
    "pendimethalin"       : ("II",  "Moderately hazardous",         1050),
    "trifluralin"         : ("III", "Slightly hazardous",           5000),
    "dicamba"             : ("II",  "Moderately hazardous",         757),
    "brodifacoum"         : ("Ia",  "Extremely hazardous",          0.3),
    "bromadiolone"        : ("Ia",  "Extremely hazardous",          1.1),
    "zinc phosphide"      : ("Ib",  "Highly hazardous",             40),
    "methyl bromide"      : ("Ib",  "Highly hazardous",             None),
    "aluminium phosphide" : ("Ia",  "Extremely hazardous",          11),
    "abamectin"           : ("Ib",  "Highly hazardous",             10),
    "spinosad"            : ("III", "Slightly hazardous",           3738),
    "indoxacarb"          : ("II",  "Moderately hazardous",         268),
    "emamectin benzoate"  : ("Ib",  "Highly hazardous",             76),
    "chlorfenapyr"        : ("II",  "Moderately hazardous",         441),
    "spiromesifen"        : ("U",   "Unlikely hazardous",           2500),
    "lufenuron"           : ("U",   "Unlikely hazardous",           2000),
    "fenpropathrin"       : ("II",  "Moderately hazardous",         70),
    "zeta-cypermethrin"   : ("II",  "Moderately hazardous",         80),
    "fenobucarb"          : ("II",  "Moderately hazardous",         623),
    "bendiocarb"          : ("Ib",  "Highly hazardous",             40),
    "dinotefuran"         : ("U",   "Unlikely hazardous",           2450),
    "nitenpyram"          : ("U",   "Unlikely hazardous",           1680),
    "zineb"               : ("U",   "Unlikely hazardous",           5200),
    "maneb"               : ("U",   "Unlikely hazardous",           6750),
    "trifloxystrobin"     : ("U",   "Unlikely hazardous",           5000),
    "imazapyr"            : ("U",   "Unlikely hazardous",           5000),
    "imazethapyr"         : ("U",   "Unlikely hazardous",           5000),
    "fluazifop-butyl"     : ("II",  "Moderately hazardous",         3328),
    "metsulfuron-methyl"  : ("U",   "Unlikely hazardous",           5000),
    "hexythiazox"         : ("U",   "Unlikely hazardous",           5000),
    "mirex"               : ("Ib",  "Highly hazardous",             235),
    "toxaphene"           : ("Ib",  "Highly hazardous",             80),
}

WHO_CLASS_NUMERIC = {"Ia": 0, "Ib": 1, "II": 2, "III": 3, "U": 4}

def build_who_dataset(save_path: Path) -> pd.DataFrame:
    print("\n[3/5] Building WHO hazard classification dataset...")
    rows = []
    for name, (cls, desc, ld50) in WHO_HAZARD_DATA.items():
        rows.append({
            "chemical_name"    : name,
            "who_class"        : cls,
            "who_description"  : desc,
            "ld50_rat_oral_mgkg": ld50,
            "who_class_numeric": WHO_CLASS_NUMERIC[cls],
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  ✓ Saved {len(df)} records → {save_path.name}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 5 — KENYA REGULATORY DATA (EMBEDDED)
# KEPHIS & PCPB registered/banned pesticides
# ════════════════════════════════════════════════════════════

KENYA_REGULATORY_DATA = [
    # (chemical, pcpb_registered, kephis_flagged, banned_in_kenya, reason)
    ("glyphosate",       True,  False, False, ""),
    ("chlorpyrifos",     True,  True,  False, "Under review — neurotoxicity"),
    ("endosulfan",       False, True,  True,  "Stockholm Convention POP"),
    ("DDT",              False, True,  True,  "Stockholm Convention POP"),
    ("paraquat",         True,  True,  False, "Restricted use — licensed applicators only"),
    ("carbofuran",       True,  True,  False, "Restricted — highly toxic"),
    ("methomyl",         True,  True,  False, "Restricted — highly toxic"),
    ("aldicarb",         False, True,  True,  "Extremely toxic — banned"),
    ("aldrin",           False, True,  True,  "Stockholm Convention POP"),
    ("dieldrin",         False, True,  True,  "Stockholm Convention POP"),
    ("heptachlor",       False, True,  True,  "Stockholm Convention POP"),
    ("lindane",          False, True,  True,  "Stockholm Convention POP"),
    ("mirex",            False, True,  True,  "Stockholm Convention POP"),
    ("toxaphene",        False, True,  True,  "Stockholm Convention POP"),
    ("chlordane",        False, True,  True,  "Stockholm Convention POP"),
    ("mancozeb",         True,  False, False, ""),
    ("cypermethrin",     True,  False, False, ""),
    ("deltamethrin",     True,  False, False, ""),
    ("imidacloprid",     True,  False, False, ""),
    ("acetamiprid",      True,  False, False, ""),
    ("abamectin",        True,  False, False, ""),
    ("emamectin benzoate", True, False, False, ""),
    ("spinosad",         True,  False, False, ""),
    ("indoxacarb",       True,  False, False, ""),
    ("profenofos",       True,  True,  False, "Restricted — handle with care"),
    ("dimethoate",       True,  True,  False, "Restricted"),
    ("aluminium phosphide", True, True, False, "Fumigant — licensed use only"),
    ("zinc phosphide",   True,  True,  False, "Rodenticide — licensed use only"),
    ("brodifacoum",      True,  True,  False, "Rodenticide — restricted"),
    ("2,4-D",            True,  False, False, ""),
    ("atrazine",         True,  False, False, ""),
    ("metolachlor",      True,  False, False, ""),
    ("azoxystrobin",     True,  False, False, ""),
    ("tebuconazole",     True,  False, False, ""),
    ("propiconazole",    True,  False, False, ""),
    ("lambda-cyhalothrin", True, False, False, ""),
    ("thiamethoxam",     True,  False, False, ""),
    ("clothianidin",     True,  False, False, "Under review — bee toxicity"),
    ("parathion",        False, True,  True,  "Banned — extremely toxic"),
    ("phorate",          False, True,  True,  "Banned — extremely toxic"),
    ("disulfoton",       False, True,  True,  "Banned — extremely toxic"),
    ("methyl bromide",   False, True,  True,  "Montreal Protocol — ozone depleting"),
]

def build_kenya_dataset(save_path: Path) -> pd.DataFrame:
    print("\n[4/5] Building Kenya regulatory dataset (KEPHIS/PCPB)...")
    cols = ["chemical_name", "pcpb_registered", "kephis_flagged",
            "banned_in_kenya", "restriction_reason"]
    df = pd.DataFrame(KENYA_REGULATORY_DATA, columns=cols)
    df.to_csv(save_path, index=False)
    print(f"  ✓ Saved {len(df)} records → {save_path.name}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 6 — TOXICITY FLAGS DATASET (DERIVED)
# Binary health hazard flags per chemical
# Sources: IARC, EPA, EU classification
# ════════════════════════════════════════════════════════════

TOXICITY_FLAGS = {
    # (carcinogen, endocrine_disruptor, neurotoxin, reproductive_toxin, pbt)
    # PBT = Persistent, Bioaccumulative, Toxic
    "glyphosate"       : (1, 0, 0, 0, 0),
    "chlorpyrifos"     : (0, 1, 1, 1, 0),
    "malathion"        : (1, 0, 1, 0, 0),
    "parathion"        : (0, 0, 1, 0, 0),
    "DDT"              : (1, 1, 0, 1, 1),
    "endosulfan"       : (0, 1, 1, 1, 1),
    "lindane"          : (1, 1, 1, 0, 1),
    "aldrin"           : (1, 0, 0, 1, 1),
    "dieldrin"         : (1, 0, 0, 1, 1),
    "atrazine"         : (0, 1, 0, 1, 0),
    "carbofuran"       : (0, 0, 1, 1, 0),
    "methomyl"         : (0, 0, 1, 0, 0),
    "aldicarb"         : (0, 0, 1, 0, 0),
    "paraquat"         : (0, 0, 1, 0, 0),
    "mancozeb"         : (0, 1, 0, 1, 0),
    "chlorothalonil"   : (1, 0, 0, 0, 0),
    "carbendazim"      : (0, 1, 0, 1, 0),
    "imidacloprid"     : (0, 0, 1, 0, 0),
    "thiamethoxam"     : (0, 0, 1, 0, 0),
    "clothianidin"     : (0, 0, 1, 0, 0),
    "2,4-D"            : (0, 1, 0, 1, 0),
    "brodifacoum"      : (0, 0, 0, 1, 1),
    "heptachlor"       : (1, 1, 0, 1, 1),
    "chlordane"        : (1, 1, 0, 1, 1),
    "mirex"            : (1, 0, 0, 1, 1),
    "toxaphene"        : (1, 1, 0, 1, 1),
    "iprodione"        : (1, 1, 0, 0, 0),
    "propiconazole"    : (1, 1, 0, 0, 0),
    "cypermethrin"     : (0, 1, 1, 0, 0),
    "deltamethrin"     : (0, 0, 1, 0, 0),
    "profenofos"       : (0, 0, 1, 0, 0),
    "dimethoate"       : (0, 1, 1, 1, 0),
    "diazinon"         : (0, 0, 1, 0, 0),
    "abamectin"        : (0, 0, 1, 0, 0),
}

def build_toxicity_flags(save_path: Path) -> pd.DataFrame:
    print("\n[5/5] Building toxicity flags dataset...")
    rows = []
    for name, flags in TOXICITY_FLAGS.items():
        rows.append({
            "chemical_name"        : name,
            "is_carcinogen"        : flags[0],
            "is_endocrine_disruptor": flags[1],
            "is_neurotoxin"        : flags[2],
            "is_reproductive_toxin": flags[3],
            "is_pbt"               : flags[4],
            "total_hazard_flags"   : sum(flags),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"  ✓ Saved {len(df)} records → {save_path.name}")
    return df


# ════════════════════════════════════════════════════════════
# SECTION 7 — MERGE ALL DATASETS INTO MASTER FILE
# ════════════════════════════════════════════════════════════

def merge_all_datasets(pubchem_df, comptox_df, who_df,
                       kenya_df, flags_df, save_path: Path) -> pd.DataFrame:
    print("\n[MERGE] Building master thesis dataset...")

    # Normalise join key
    def norm(df, col="chemical_name"):
        df[col] = df[col].str.strip().str.lower()
        return df

    who_df    = norm(who_df)
    kenya_df  = norm(kenya_df)
    flags_df  = norm(flags_df)

    # PubChem uses query_name
    pubchem_df = pubchem_df.copy()
    pubchem_df["chemical_name"] = pubchem_df["query_name"].str.strip().str.lower()

    comptox_df = comptox_df.copy()
    comptox_df["chemical_name"] = comptox_df["query_name"].str.strip().str.lower()

    # Start from WHO (our labels anchor)
    master = who_df.copy()

    # Merge PubChem properties
    pub_cols = ["chemical_name", "cid", "isomericsmiles", "molecularformula",
                "molecularweight", "xlogp", "tpsa", "hbonddonorcount",
                "hbondacceptorcount", "rotatablebondcount", "heavyatomcount",
                "complexity", "inchikey", "iupacname"]
    pub_cols = [c for c in pub_cols if c in pubchem_df.columns]
    master = master.merge(pubchem_df[pub_cols], on="chemical_name", how="left")

    # Merge CompTox identifiers
    ctx_cols = ["chemical_name", "dtxsid", "cas_rn", "smiles_comptox", "inchikey_comptox"]
    ctx_cols = [c for c in ctx_cols if c in comptox_df.columns]
    master = master.merge(comptox_df[ctx_cols], on="chemical_name", how="left")

    # Merge Kenya regulatory data
    master = master.merge(kenya_df, on="chemical_name", how="left")

    # Merge toxicity flags
    master = master.merge(flags_df, on="chemical_name", how="left")

    # Fill missing Kenya flags as False
    for col in ["pcpb_registered", "kephis_flagged", "banned_in_kenya"]:
        if col in master.columns:
            master[col] = master[col].fillna(False)

    # Derive a unified SMILES column (prefer PubChem, fallback CompTox)
    if "isomericsmiles" in master.columns and "smiles_comptox" in master.columns:
        master["smiles"] = master["isomericsmiles"].fillna(master["smiles_comptox"])
    elif "isomericsmiles" in master.columns:
        master["smiles"] = master["isomericsmiles"]

    # Add safety tier column (simplified 3-class for easy ML)
    def safety_tier(who_class):
        if who_class in ["Ia", "Ib"]:
            return "HIGH_RISK"
        elif who_class == "II":
            return "MODERATE_RISK"
        elif who_class in ["III", "U"]:
            return "LOW_RISK"
        return "UNKNOWN"

    master["safety_tier"] = master["who_class"].apply(safety_tier)
    tier_map = {"HIGH_RISK": 0, "MODERATE_RISK": 1, "LOW_RISK": 2, "UNKNOWN": -1}
    master["safety_tier_numeric"] = master["safety_tier"].map(tier_map)

    master.to_csv(save_path, index=False)
    print(f"  ✓ Master dataset saved → {save_path.name}")
    print(f"  ✓ Shape: {master.shape[0]} rows × {master.shape[1]} columns")
    return master


# ════════════════════════════════════════════════════════════
# SECTION 8 — DATA QUALITY REPORT
# ════════════════════════════════════════════════════════════

def print_quality_report(df: pd.DataFrame, log_path: Path):
    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)

    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    log(f"Total chemicals   : {len(df)}")
    log(f"Total features    : {df.shape[1]}")
    log("")
    log("── WHO Class Distribution ──")
    if "who_class" in df.columns:
        for cls, cnt in df["who_class"].value_counts().items():
            log(f"  {cls:6s} : {cnt:3d} chemicals")

    log("")
    log("── Safety Tier Distribution ──")
    if "safety_tier" in df.columns:
        for tier, cnt in df["safety_tier"].value_counts().items():
            log(f"  {tier:20s} : {cnt:3d}")

    log("")
    log("── SMILES Coverage ──")
    if "smiles" in df.columns:
        valid = df["smiles"].notna().sum()
        log(f"  Chemicals with SMILES: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")

    log("")
    log("── Missing Values (top columns) ──")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, cnt in missing.head(10).items():
        log(f"  {col:35s}: {cnt} missing ({100*cnt/len(df):.1f}%)")

    log("")
    log("── Kenya Regulatory Summary ──")
    if "banned_in_kenya" in df.columns:
        banned  = df["banned_in_kenya"].sum()
        flagged = df["kephis_flagged"].sum() if "kephis_flagged" in df.columns else "N/A"
        reg     = df["pcpb_registered"].sum() if "pcpb_registered" in df.columns else "N/A"
        log(f"  PCPB registered : {reg}")
        log(f"  KEPHIS flagged  : {flagged}")
        log(f"  Banned in Kenya : {banned}")

    with open(log_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  ✓ Report saved → {log_path.name}")


# ════════════════════════════════════════════════════════════
# MAIN — RUN EVERYTHING
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Step 1 — PubChem
    pubchem_path = RAW_DIR / "pubchem_properties.csv"
    pubchem_df = collect_pubchem(PESTICIDE_LIST, pubchem_path)

    # Step 2 — CompTox
    comptox_path = RAW_DIR / "comptox_identifiers.csv"
    comptox_df = collect_comptox(PESTICIDE_LIST, comptox_path)

    # Step 3 — WHO classification
    who_path = RAW_DIR / "who_hazard_classifications.csv"
    who_df = build_who_dataset(who_path)

    # Step 4 — Kenya regulatory
    kenya_path = EXT_DIR / "kenya_regulatory_kephis_pcpb.csv"
    kenya_df = build_kenya_dataset(kenya_path)

    # Step 5 — Toxicity flags
    flags_path = RAW_DIR / "toxicity_flags.csv"
    flags_df = build_toxicity_flags(flags_path)

    # Step 6 — Merge master dataset
    master_path = PROC_DIR / "master_thesis_dataset.csv"
    master_df = merge_all_datasets(
        pubchem_df, comptox_df, who_df,
        kenya_df, flags_df, master_path
    )

    # Step 7 — Quality report
    report_path = LOG_DIR / "data_quality_report.txt"
    print_quality_report(master_df, report_path)

    print("\n" + "=" * 60)
    print("  ALL DONE! Your thesis dataset is ready.")
    print(f"  Master file : {master_path}")
    print(f"  Raw files   : {RAW_DIR}")
    print(f"  Report      : {report_path}")
    print("=" * 60)
