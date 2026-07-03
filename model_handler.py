"""
Virus Prediction Model Handler Module
Encapsulates TabularResNet model loading, feature preprocessing, and prediction logic
Uses PyTorch-based neural networks with bundled preprocessing in .pth files
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================================
# VIRUS & SYMPTOM MAPPINGS
# ============================================================================


# Main virus mapping loaded from CSV at runtime.
DEFAULT_VIRUS_MAPPING = {
    0: 'Chikungunya Virus',
    1: 'Dengue Virus',
    2: 'Enterovirus',
    3: 'Hepatitis A Virus',
    4: 'Hepatitis B Virus',
    5: 'Hepatitis C Virus',
    6: 'Hepatitis E Virus',
    7: 'Herpes Simplex Virus (HSV)',
    8: 'Influenza A H1N1',
    9: 'Influenza A H3N2',
    10: 'Influenza B Victoria',
    11: 'Japanese Encephalitis',
    12: 'Leptospira',
    13: 'Measles Virus',
    14: 'Mumps Virus',
    15: 'Other_Viruses',
    16: 'Parvovirus',
    17: 'Respiratory Adenovirus',
    18: 'Respiratory Syncytial Virus (RSV)',
    19: 'Rotavirus',
    20: 'Rubella',
    21: 'SARS-Cov-2',
    22: 'Scrub typhus (Orientia tsutsugamushi)',
    23: 'Varicella zoster virus (VZV)',
}

# Other Virus sub-classification mapping loaded from CSV at runtime.
DEFAULT_OTHER_VIRUS_MAPPING = {
    0: 'Human papillomavirus (HPV)',
    1: 'Kyasanur Forest Disease',
    2: 'Metapneumovirus',
    3: 'Norovirus',
    4: 'Other Influenza',
    5: 'Rhinovirus',
    6: 'West Nile virus (WNV)',
    7: 'Zika',
}

DEFAULT_SYNDROME_MAPPING = {
    0: 'ARI/Influenza Like Illness (ILI)',
    1: 'Acute Diarrheal Disease',
    2: 'Acute Encephalitis Syndrome (AES)',
    3: 'Conjunctivitis',
    4: 'Fever with Rash',
    5: 'Hemorrhagic fever',
    6: 'Jaundice of < 4 weeks',
    7: 'Only Fever < 7 days',
    8: 'Severe Acute Respiratory Infection (SARI)',
}

VIRUS_MAPPING = dict(DEFAULT_VIRUS_MAPPING)
OTHER_VIRUS_MAPPING = dict(DEFAULT_OTHER_VIRUS_MAPPING)
COMBINED_VIRUS_MAPPING = {}
SYNDROME_MAPPING = dict(DEFAULT_SYNDROME_MAPPING)
# Maps Overall_Syndromes -> Encoded_Value for UI display
SYNDROME_DISPLAY_MAPPING = {}


def _try_read_csv(csv_path, encodings=['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']):
    """Try to read CSV with multiple encodings to handle encoding issues."""
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(
        f"Could not read CSV {csv_path} with any of the attempted encodings: {encodings}")


def _read_label_mapping_csv(csv_path, label_column, encoded_column):
    df = _try_read_csv(csv_path)
    required_cols = {label_column, encoded_column}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Invalid mapping file: {csv_path}. Expected columns: {required_cols}."
        )

    df = df.dropna(subset=[label_column, encoded_column])
    df[encoded_column] = df[encoded_column].astype(int)
    return dict(zip(df[encoded_column], df[label_column].astype(str)))


def _read_syndrome_display_mapping_csv(csv_path):
    """Load syndrome display mapping from SyndromeMapping.csv
    Returns: dict mapping Overall_Syndromes -> Encoded_Value
    """
    df = _try_read_csv(csv_path)
    required_cols = {'Overall_Syndromes', 'Encoded_Value'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Invalid syndrome mapping file: {csv_path}. Expected columns: {required_cols}."
        )

    df = df.dropna(subset=['Overall_Syndromes', 'Encoded_Value'])
    df['Encoded_Value'] = df['Encoded_Value'].astype(int)
    # Remove duplicates by keeping first occurrence (they map to same encoded value anyway)
    return dict(zip(df['Overall_Syndromes'].astype(str), df['Encoded_Value']))


def refresh_virus_mappings(
    major_csv_path=None,
    other_csv_path=None,
    syndrome_csv_path=None,
):
    """
    Reload virus name mappings from CSV files and update in place.

    Args:
        major_csv_path: Path to encoding_major_VIRUS_NAME.csv
        other_csv_path: Path to encoding_other_VIRUS_NAME.csv
    """
    base_dir = Path(__file__).resolve().parent
    major_csv_path = major_csv_path or base_dir / "encoding_major_VIRUS_NAME.csv"
    other_csv_path = other_csv_path or base_dir / "encoding_other_VIRUS_NAME.csv"
    syndrome_csv_path = syndrome_csv_path or base_dir / "SyndromeMapping.csv"

    major_mapping = dict(DEFAULT_VIRUS_MAPPING)
    other_mapping = dict(DEFAULT_OTHER_VIRUS_MAPPING)
    syndrome_mapping = dict(DEFAULT_SYNDROME_MAPPING)

    try:
        if Path(major_csv_path).exists():
            major_mapping = _read_label_mapping_csv(
                major_csv_path, "Original", "Encoded")
        else:
            st.warning(f"Major mapping file not found: {major_csv_path}")
    except Exception as exc:
        st.warning(f"Failed to load major mapping from CSV: {exc}")

    try:
        if Path(other_csv_path).exists():
            other_mapping = _read_label_mapping_csv(
                other_csv_path, "Original", "Encoded")
        else:
            st.warning(f"Other mapping file not found: {other_csv_path}")
    except Exception as exc:
        st.warning(f"Failed to load other mapping from CSV: {exc}")

    # Load syndrome display mapping (Overall_Syndromes -> Encoded_Value)
    syndrome_display_mapping = dict()
    try:
        if Path(syndrome_csv_path).exists():
            syndrome_display_mapping = _read_syndrome_display_mapping_csv(
                syndrome_csv_path)
            # Also extract Syndrome_Label -> Encoded_Value for backward compatibility
            df = _try_read_csv(syndrome_csv_path)
            df = df.dropna(subset=['Syndrome_Label', 'Encoded_Value'])
            df['Encoded_Value'] = df['Encoded_Value'].astype(int)
            syndrome_mapping = dict(
                zip(df['Encoded_Value'], df['Syndrome_Label'].astype(str)))
            syndrome_mapping = dict.fromkeys(
                syndrome_mapping.values(), -1)  # Reset
            for idx, label in zip(df['Encoded_Value'], df['Syndrome_Label'].astype(str)):
                syndrome_mapping[idx] = label
        else:
            st.warning(f"Syndrome mapping file not found: {syndrome_csv_path}")
    except Exception as exc:
        st.warning(f"Failed to load syndrome mapping from CSV: {exc}")

    VIRUS_MAPPING.clear()
    VIRUS_MAPPING.update(major_mapping)

    OTHER_VIRUS_MAPPING.clear()
    OTHER_VIRUS_MAPPING.update(other_mapping)

    SYNDROME_MAPPING.clear()
    SYNDROME_MAPPING.update(syndrome_mapping)

    SYNDROME_DISPLAY_MAPPING.clear()
    SYNDROME_DISPLAY_MAPPING.update(syndrome_display_mapping)

    COMBINED_VIRUS_MAPPING.clear()
    COMBINED_VIRUS_MAPPING.update(
        {f"main_{k}": v for k, v in VIRUS_MAPPING.items() if k != 15}
    )
    COMBINED_VIRUS_MAPPING.update(
        {f"other_{k}": f"Other Viruses → {v}" for k,
            v in OTHER_VIRUS_MAPPING.items()}
    )


# Combined virus mapping for validation dropdown
refresh_virus_mappings()

# All clinical symptoms (no spaces to match training data)
ALL_SYMPTOMS = [
    'ABDOMINALPAIN', 'ALTEREDSENSORIUM', 'ARTHRALGIA', 'BREATHLESSNESS', 'BULLAE', 'CHILLS', 'COUGH', 
    'CRUSHINGEYES', 'DARKURINE', 'DIARRHEA', 'DISCHARGEEYES', 'DYSENTERY', 'ESCHAR', 'FEVER', 'HEADACHE', 
    'HEPATOMEGALY', 'IRRITABLITY', 'JAUNDICE', 'MACULOPAPULARRASH', 'MALAISE', 'MUSCULARRASH', 'MYALGIA', 
    'NAUSEA', 'NECKRIGIDITY', 'PAPULARRASH', 'PUSTULARRASH', 'REDEYE', 'RETROORBITALPAIN', 'RHINORRHEA', 
    'RIGORS', 'SEIZURES', 'SOMNOLENCE', 'SORETHROAT', 'SWELLINGEYES', 'VOMITING'
]


# ============================================================================
# SYNDROME -> EXCLUDED VIRUS RULES
# ============================================================================
# Source: the "Syndrome / Viruses That Should Not Be Present" clinical
# reference table. For a given syndrome, any virus listed here is considered
# clinically implausible and is removed from that syndrome's predictions
# before the top-N ranking is built — the next-highest-confidence,
# syndrome-consistent prediction moves up to take its place.
#
# Keyed by Syndrome_encoded (0-8, matches DEFAULT_SYNDROME_MAPPING and the
# 'Syndrome_encoded' / 'syndrome' fields in patient_data). Names are matched
# case-insensitively against VIRUS_MAPPING / OTHER_VIRUS_MAPPING after
# passing through VIRUS_NAME_ALIASES, so this table doesn't need to track
# every spelling variant used elsewhere in the codebase. Names with no match
# in either mapping (e.g. "HIV", "Haemophilus influenzae", "Toxoplasma",
# "Unknown" — not classes either model can predict) are harmless no-ops; kept
# here so this table mirrors the source reference table exactly.
SYNDROME_EXCLUDED_VIRUSES = {
    0: {  # ARI/Influenza Like Illness (ILI)
        'Dengue Virus', 'HIV', 'Haemophilus influenzae', 'Hepatitis A Virus',
        'Hepatitis B Virus', 'Hepatitis C Virus', 'Hepatitis E Virus',
        'Herpes Simplex Virus (HSV)', 'Human papillomavirus (HPV)',
        'Japanese Encephalitis', 'Kyasanur Forest Disease', 'Leptospira',
        'Norovirus', 'Rotavirus', 'Rubella',
        'Scrub typhus (Orientia tsutsugamushi)', 'Toxoplasma', 'Unknown',
        'West Nile virus (WNV)',
    },
    1: {  # Acute Diarrheal Disease
        'HIV', 'Haemophilus influenzae', 'Hepatitis B Virus',
        'Hepatitis C Virus', 'Herpes Simplex Virus (HSV)',
        'Human papillomavirus (HPV)', 'Japanese Encephalitis',
        'Kyasanur Forest Disease', 'Mumps Virus', 'Parvovirus', 'Rubella',
        'Toxoplasma', 'Varicella zoster virus (VZV)', 'West Nile virus (WNV)',
    },
    2: {  # Acute Encephalitis Syndrome (AES)
        'HIV', 'Haemophilus influenzae', 'Hepatitis A Virus',
        'Hepatitis B Virus', 'Hepatitis C Virus', 'Hepatitis E Virus',
        'Human papillomavirus (HPV)', 'Influenza A H1N1', 'Influenza A H3N2',
        'Influenza B Victoria', 'Metapneumovirus', 'Norovirus',
        'Other Influenza', 'Respiratory Adenovirus',
        'Respiratory Syncytial Virus (RSV)', 'Rhinovirus', 'Toxoplasma',
    },
    3: {  # Conjunctivitis
        'HIV', 'Haemophilus influenzae', 'Hepatitis A Virus',
        'Hepatitis B Virus', 'Hepatitis C Virus', 'Leptospira', 'Mumps Virus',
        'Respiratory Syncytial Virus (RSV)',
        'Scrub typhus (Orientia tsutsugamushi)', 'Toxoplasma',
    },
    4: {  # Fever with Rash
        'HIV', 'Haemophilus influenzae', 'Hepatitis A Virus',
        'Hepatitis B Virus', 'Hepatitis C Virus', 'Hepatitis E Virus',
        'Influenza A H1N1', 'Influenza A H3N2', 'Influenza B Victoria',
        'Respiratory Syncytial Virus (RSV)', 'Rotavirus', 'SARS-Cov-2',
        'Toxoplasma',
    },
    5: {  # Hemorrhagic fever
        'Enterovirus', 'HIV', 'Haemophilus influenzae', 'Hepatitis A Virus',
        'Hepatitis E Virus', 'Herpes Simplex Virus (HSV)', 'Measles Virus',
        'Mumps Virus', 'Norovirus', 'Parvovirus', 'Respiratory Adenovirus',
        'Respiratory Syncytial Virus (RSV)', 'Rubella', 'Toxoplasma',
    },
    6: {  # Jaundice of < 4 weeks
        'Chikungunya Virus', 'Enterovirus', 'HIV', 'Haemophilus influenzae',
        'Human papillomavirus (HPV)', 'Influenza A H1N1', 'Influenza A H3N2',
        'Influenza B Victoria', 'Measles Virus', 'Mumps Virus', 'Parvovirus',
        'Respiratory Adenovirus', 'Respiratory Syncytial Virus (RSV)',
        'Rhinovirus', 'Rotavirus', 'Rubella', 'Toxoplasma',
    },
    7: {  # Only Fever < 7 days
        'HIV', 'Haemophilus influenzae', 'Toxoplasma',
    },
    8: {  # Severe Acute Respiratory Infection (SARI)
        'Dengue Virus', 'HIV', 'Haemophilus influenzae', 'Hepatitis A Virus',
        'Hepatitis B Virus', 'Hepatitis C Virus', 'Hepatitis E Virus',
        'Herpes Simplex Virus (HSV)', 'Human papillomavirus (HPV)',
        'Japanese Encephalitis', 'Kyasanur Forest Disease', 'Leptospira',
        'Norovirus', 'Other Influenza', 'Rotavirus', 'Rubella',
        'Scrub typhus (Orientia tsutsugamushi)', 'Toxoplasma',
        'West Nile virus (WNV)',
    },
}

# A couple of reference-table entries use different wording than the model's
# own class names (e.g. the table says "Respiratory Adenovirus", but
# DEFAULT_VIRUS_MAPPING's class 17 may load as plain "Adenovirus" from
# encoding_major_VIRUS_NAME.csv). Both spellings are listed so matching is
# correct regardless of which one is currently in VIRUS_MAPPING.
VIRUS_NAME_ALIASES = {
    'respiratory adenovirus': 'adenovirus',
    'adenovirus': 'adenovirus',
}


def _normalize_virus_name(name):
    """Lowercase + alias-normalize a virus name for exclusion-list matching."""
    key = str(name).strip().lower()
    return VIRUS_NAME_ALIASES.get(key, key)


def _build_excluded_index_set(syndrome_encoded, virus_mapping):
    """
    Resolve a syndrome's excluded-virus names (SYNDROME_EXCLUDED_VIRUSES) to
    the set of class indices in `virus_mapping` (VIRUS_MAPPING or
    OTHER_VIRUS_MAPPING) that should not appear for that syndrome. Returns an
    empty set if the syndrome is unrecognised or has no listed exclusions.
    """
    if syndrome_encoded is None:
        return set()
    try:
        syndrome_key = int(syndrome_encoded)
    except (TypeError, ValueError):
        return set()
    excluded_names = SYNDROME_EXCLUDED_VIRUSES.get(syndrome_key, set())
    if not excluded_names:
        return set()
    excluded_normalized = {_normalize_virus_name(n) for n in excluded_names}
    return {
        idx for idx, name in virus_mapping.items()
        if _normalize_virus_name(name) in excluded_normalized
    }


def _filter_topk(y_pred_proba, excluded_indices, k=5):
    """
    Rank all classes in y_pred_proba by probability (descending), drop any
    class in `excluded_indices`, and return the top-k indices of what's left.

    May return fewer than k indices if fewer than k syndrome-consistent
    classes exist (this happens for a few syndrome / Other-Viruses-model
    combinations, where most of the 8 sub-categories get excluded) — a virus
    the reference table marks as "should not appear" is never padded back in
    just to fill out a 5-item list.
    """
    full_ranking = np.argsort(y_pred_proba)[::-1]
    if not excluded_indices:
        return full_ranking[:k]
    filtered = np.array(
        [idx for idx in full_ranking if idx not in excluded_indices])
    if filtered.size == 0:
        # Every class excluded for this syndrome — shouldn't happen with the
        # current reference table, but fail safe to the raw ranking rather
        # than return an empty result.
        return full_ranking[:k]
    return filtered[:k]


# ============================================================================
# DEVICE DETECTION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# GRTT ARCHITECTURE (updated runtime model)
# ============================================================================


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(a * F.gelu(b))


class TransformerBlock(nn.Module):
    """Transformer block with gated residual connections"""

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = GEGLU(d_model, d_ff)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + torch.sigmoid(self.attn_gate) * self.drop(attn_out)

        h = self.ln2(x)
        ff_out = self.ff(h)
        x = x + torch.sigmoid(self.ff_gate) * self.drop(ff_out)
        return x


class TabularResNet(nn.Module):
    """
    Updated runtime architecture for the virus classifier.

    The class name is preserved for compatibility with the existing app and
    older checkpoints, but the implementation matches the updated GRTT design.
    """

    def __init__(
        self,
        num_binary,
        num_continuous,
        cat_dims,
        num_classes,
        d_token: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        token_drop: float = 0.05,
    ):
        super().__init__()

        self.num_cat = len(cat_dims)
        self.num_continuous = num_continuous
        self.token_drop = token_drop

        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(card, emb) for card, emb in cat_dims])
        self.cat_proj = nn.ModuleList(
            [nn.Linear(emb, d_token) for _, emb in cat_dims])
        self.cont_proj = nn.ModuleList(
            [nn.Linear(1, d_token) for _ in range(num_continuous)])
        self.cont_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(d_token)) for _ in range(num_continuous)])

        self.bin_linear = nn.Linear(num_binary, d_token)
        self.bin_gate = nn.Parameter(torch.zeros(1))

        self.max_tokens = 1 + self.num_cat + \
            num_continuous + (1 if num_binary > 0 else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, d_token))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_token, n_heads=n_heads,
                             d_ff=d_token * 2, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_token)
        self.proj_head = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
            nn.ReLU(inplace=True),
            nn.Linear(d_token, 128),
        )
        self.head = nn.Linear(d_token, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xb, xc, xcat, return_embed=False):
        B = xb.size(0)
        tokens = []

        for i in range(self.num_cat):
            cat_emb = self.cat_embeds[i](xcat[:, i])
            tokens.append(self.cat_proj[i](cat_emb).unsqueeze(1))

        for i in range(self.num_continuous):
            cont_token = self.cont_proj[i](xc[:, i:i + 1]) * self.cont_scale[i]
            tokens.append(cont_token.unsqueeze(1))

        if xb.size(1) > 0:
            bin_emb = torch.sigmoid(self.bin_gate) * self.bin_linear(xb)
            tokens.append(bin_emb.unsqueeze(1))

        x = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.training and self.token_drop > 0.0:
            keep = torch.rand(x.size(1), device=x.device) > self.token_drop
            keep[0] = True
            x = x[:, keep, :]
            pos_embed_used = self.pos_embed[:, :self.max_tokens, :][:, keep, :]
        else:
            pos_embed_used = self.pos_embed[:, :x.size(1), :]

        x = x + pos_embed_used

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        pooled = 0.7 * x[:, 0] + 0.3 * x[:, 1:].mean(dim=1)

        if return_embed:
            z = F.normalize(self.proj_head(pooled), dim=1)
            return pooled, z

        return self.head(pooled)


GRTT = TabularResNet


# ============================================================================
# VIRUS PREDICTOR CLASS (using TabularResNet)
# ============================================================================

class VirusPredictor:
    """
    Encapsulates TabularResNet model loading, feature preprocessing, and prediction.
    Loads bundled .pth files with model weights + preprocessing objects.
    """

    def __init__(self, model1_path='models/grtt_major_production.pth',
                 model2_path='models/grtt_other_production.pth'):
        """
        Initialize predictor by loading both pretrained models.

        Args:
            model1_path: Path to primary model .pth file (26 major viruses)
            model2_path: Path to secondary model .pth file (13 other virus sub-types)
        """
        self.model1 = None
        self.model2 = None
        self.preprocessing1 = None
        self.preprocessing2 = None
        self.model_info1 = {}
        self.model_info2 = {}
        self.load_models(model1_path, model2_path)

    def load_models(self, model1_path, model2_path):
        """
        Load both TabularResNet models with bundled preprocessing.

        Args:
            model1_path: Path to primary model
            model2_path: Path to secondary model

        Returns:
            bool: True if both models loaded successfully
        """
        try:
            allowlisted = [
                SimpleImputer,
                StandardScaler,
                LabelEncoder,
                np.core.multiarray._reconstruct,
            ]
            torch.serialization.add_safe_globals(allowlisted)

            def _safe_torch_load(path):
                with torch.serialization.safe_globals(allowlisted):
                    try:
                        return torch.load(path, map_location=DEVICE, weights_only=True)
                    except Exception:
                        # st.warning(
                        #     "Safe model load failed; retrying with weights_only=False."
                        #     "Only do this if you trust the checkpoint source."
                        # )
                        return torch.load(path, map_location=DEVICE, weights_only=False)

            checkpoint1 = _safe_torch_load(model1_path)
            self.model1, self.preprocessing1, self.model_info1 = self._load_single_model_bundle(
                checkpoint1)

            checkpoint2 = _safe_torch_load(model2_path)
            self.model2, self.preprocessing2, self.model_info2 = self._load_single_model_bundle(
                checkpoint2)
            # Warn if syndrome feature is not present in categorical columns of either model

            def _has_syndrome_feature(preproc):
                cat_cols = preproc.get('cat_cols', []) if preproc else []
                for name in cat_cols:
                    if 'syndrom' in name.lower() or 'syndrome' in name.lower():
                        return True
                return False

            if not _has_syndrome_feature(self.preprocessing1):
                st.warning(
                    "Primary model loaded does not include a syndrome categorical column — syndrome will not be used by this model unless retrained with that feature.")
            if not _has_syndrome_feature(self.preprocessing2):
                st.info("Secondary model loaded does not include a syndrome categorical column — if you expect syndrome input for sub-classification, retrain the secondary model including that feature.")

            return True

        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}")
            return False
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False

    @staticmethod
    def _normalize_imputer_state(preprocessing):
        """Backfill SimpleImputer attributes for cross-version sklearn compatibility."""
        for key in ('imp_cont', 'imp_bin', 'imputer'):
            imputer = preprocessing.get(key)
            if isinstance(imputer, SimpleImputer) and not hasattr(imputer, '_fill_dtype'):
                if hasattr(imputer, '_fit_dtype'):
                    imputer._fill_dtype = imputer._fit_dtype
                elif hasattr(imputer, 'statistics_'):
                    imputer._fill_dtype = np.asarray(imputer.statistics_).dtype
                else:
                    imputer._fill_dtype = np.dtype('float64')

    def _load_single_model_bundle(self, checkpoint):
        """Normalize old and new checkpoint formats into a common runtime shape."""
        config = checkpoint.get('model_config')
        if config is None:
            raise ValueError('Checkpoint is missing model_config.')

        model = TabularResNet(**config).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if 'preprocessing' in checkpoint:
            preprocessing = dict(checkpoint['preprocessing'])
        else:
            preprocessing = {
                'binary_cols': checkpoint.get('binary_cols', []),
                'cont_cols': checkpoint.get('cont_cols', []),
                'cat_cols': checkpoint.get('cat_cols', []),
                'imputer': checkpoint.get('imputer'),
                'scaler': checkpoint.get('scaler'),
                'cat_encoders': checkpoint.get('cat_encoders', {}),
            }

        preprocessing.setdefault('imp_cont', preprocessing.get('imputer'))
        preprocessing.setdefault('imp_bin', None)
        preprocessing.setdefault(
            'le_dict', preprocessing.get('cat_encoders', {}))
        self._normalize_imputer_state(preprocessing)

        info = {
            'virus_mapping': checkpoint.get('virus_mapping', {}),
            'best_acc': checkpoint.get('best_acc'),
            'best_f1_macro': checkpoint.get('best_f1_macro'),
            'best_epoch': checkpoint.get('best_epoch'),
        }

        return model, preprocessing, info

    def preprocess_features(self, patient_data, preprocessing):
        """
        Transform patient data dict → binary, continuous, categorical tensors.
        Uses preprocessing objects stored in the model bundle.
        Includes feature engineering to match training data.

        Args:
            patient_data: Dictionary with patient demographics and symptoms
            preprocessing: Dict with binary_cols, cat_cols, cont_cols, scalers, encoders

        Returns:
            Tuple of (xb, xc, xcat) PyTorch tensors ready for model inference
        """
        try:
            binary_cols = preprocessing.get('binary_cols', [])
            cat_cols = preprocessing.get('cat_cols', [])
            cont_cols = preprocessing.get('cont_cols', [])
            imp_cont = preprocessing.get(
                'imp_cont') or preprocessing.get('imputer')
            scaler = preprocessing.get('scaler')
            imp_bin = preprocessing.get('imp_bin')
            le_dict = preprocessing.get(
                'le_dict') or preprocessing.get('cat_encoders', {})

            df = pd.DataFrame([patient_data])

            # --- Sex safety: models were trained on Female(0)/Male(1) only. ---
            # Map any other value (e.g. "Other" = 2) to 0 for MODEL INPUT ONLY, so the
            # network never receives an out-of-distribution value. Female/Male rows are
            # untouched -> accuracy on real data is unchanged. patient_data (and the saved
            # DB record) keep the original value, so "Other" is still recorded correctly.
            if 'SEX' in df.columns:
                df['SEX'] = df['SEX'].where(df['SEX'].isin([0, 1]), 0)

            # ========== FEATURE ENGINEERING (to match training) ==========

            # 1. AGE FEATURES
            age_median = df['age'].median() if 'age' in df.columns else 30
            df['age'] = df['age'].fillna(age_median).clip(0, 120)

            # Age groups
            age_group = pd.cut(df['age'], bins=[0, 5, 18, 45, 65, 150], labels=[
                               0, 1, 2, 3, 4]).cat.codes
            df['age_group'] = age_group.replace(-1, 2)

            # 2. SYMPTOM GROUPS & COUNTS
            symptom_cols = [col for col in ALL_SYMPTOMS if col in df.columns]
            respiratory_cols = ['COUGH', 'BREATHLESSNESS',
                                'RHINORRHEA', 'SORE THROAT']
            gi_cols = ['DIARRHEA', 'DYSENTERY',
                       'NAUSEA', 'VOMITING', 'ABDOMINAL PAIN']
            neuro_cols = ['HEADACHE', 'ALTERED SENSORIUM', 'SEIZURES',
                          'SOMNOLENCE', 'NECK RIGIDITY', 'IRRITABILITY']
            skin_cols = ['PAPULAR RASH', 'PUSTULAR RASH',
                         'MACULOPAPULAR RASH', 'BULLAE']
            systemic_cols = ['MYALGIA', 'ARTHRALGIA',
                             'CHILLS', 'RIGORS', 'MALAISE']

            for col in symptom_cols:
                df[col] = df[col].fillna(0)

            df['durationofillness'] = df.get('durationofillness', 0)
            df['durationofillness'] = df['durationofillness'].fillna(0)

            # Create symptom counts
            df['symptom_count'] = df[symptom_cols].sum(axis=1)

            # Symptom group counts (handle missing columns)
            resp_present = [c for c in respiratory_cols if c in df.columns]
            df['respiratory_symptoms'] = df[resp_present].sum(
                axis=1) if resp_present else 0

            gi_present = [c for c in gi_cols if c in df.columns]
            df['gi_symptoms'] = df[gi_present].sum(axis=1) if gi_present else 0

            neuro_present = [c for c in neuro_cols if c in df.columns]
            df['neuro_symptoms'] = df[neuro_present].sum(
                axis=1) if neuro_present else 0

            skin_present = [c for c in skin_cols if c in df.columns]
            df['skin_symptoms'] = df[skin_present].sum(
                axis=1) if skin_present else 0

            systemic_present = [c for c in systemic_cols if c in df.columns]
            df['systemic_symptoms'] = df[systemic_present].sum(
                axis=1) if systemic_present else 0

            df['symptom_diversity'] = (df[symptom_cols] > 0).sum(axis=1)

            # 3. GEO-TEMPORAL FEATURES
            if 'month' in df.columns:
                # Season mapping
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 0  # Winter
                    elif month in [3, 4, 5]:
                        return 1  # Summer
                    elif month in [6, 7, 8, 9]:
                        return 2  # Monsoon
                    else:
                        return 3  # Post-monsoon

                df['season'] = df['month'].apply(get_season)

                # Monsoon and winter flags (if not already present)
                if 'is_monsoon' not in df.columns:
                    df['is_monsoon'] = df['month'].isin(
                        [6, 7, 8, 9]).astype(int)
                if 'is_winter' not in df.columns:
                    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)

                # Cyclical encoding
                if 'month_sin' not in df.columns:
                    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                if 'month_cos' not in df.columns:
                    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

                # Create missing temporal features (approximate)
                df['week_of_year'] = df['month'] * 4  # Rough approximation
                df['day_of_year'] = df['month'] * 30   # Rough approximation
                df['quarter'] = ((df['month'] - 1) // 3) + 1

            # District encoding (create if missing)
            if 'districtencoded' in df.columns and 'district_encoded' not in df.columns:
                df['district_encoded'] = df['districtencoded']
            elif 'district_encoded' not in df.columns:
                df['district_encoded'] = 0  # Default value

            # State encoding (ensure correct column name)
            if 'labstate' in df.columns and 'lab_state' not in df.columns:
                df['lab_state'] = df['labstate']
            elif 'lab_state' not in df.columns:
                df['lab_state'] = df.get('labstate', 0)

            # Year normalization (if present)
            if 'year' in df.columns:
                # Use 2012-2026 range from training
                df['year_normalized'] = (df['year'] - 2012) / (2026 - 2012)
            else:
                df['year_normalized'] = 0.5  # Default to middle

            # 4. INTERACTION FEATURES

            # Season-symptom interactions
            if 'season' in df.columns:
                df['monsoon_respiratory'] = df.get(
                    'is_monsoon', 0) * df['respiratory_symptoms']
                df['winter_respiratory'] = df.get(
                    'is_winter', 0) * df['respiratory_symptoms']
                df['monsoon_fever'] = df.get(
                    'is_monsoon', 0) * df.get('FEVER', 0)

                # State-season interaction
                df['state_season'] = df['lab_state'] * 10 + df['season']

                # District interactions
                df['district_season'] = df['district_encoded'] * 10 + df['season']
                df['district_month'] = df['district_encoded'] * \
                    100 + df.get('month', 1)

            # State-symptom interactions
            df['state_respiratory'] = df['lab_state'] * \
                df['respiratory_symptoms']
            df['state_fever'] = df['lab_state'] * df.get('FEVER', 0)
            df['state_gi'] = df['lab_state'] * df['gi_symptoms']

            # Other interactions
            df['fever_respiratory'] = df.get(
                'FEVER', 0) * df['respiratory_symptoms']
            df['fever_gi'] = df.get('FEVER', 0) * df['gi_symptoms']
            df['fever_neuro'] = df.get('FEVER', 0) * df['neuro_symptoms']
            df['fever_skin'] = df.get('FEVER', 0) * df['skin_symptoms']
            df['fever_duration'] = df.get('FEVER', 0) * df['durationofillness']
            df['fever_headache'] = df.get('FEVER', 0) * df.get('HEADACHE', 0)
            df['fever_cough'] = df.get('FEVER', 0) * df.get('COUGH', 0)

            # Severity and complexity features
            df['severity_score'] = df['symptom_count'] * df['durationofillness']
            df['age_symptom'] = df['age'] * df['symptom_count']
            df['age_duration'] = df['age'] * df['durationofillness']
            df['patienttype_age'] = df.get('PATIENTTYPE', 1) * df['age_group']
            df['sex_respiratory'] = df.get(
                'SEX', 1) * df['respiratory_symptoms']
            df['duration_symptom_ratio'] = df['durationofillness'] / \
                (df['symptom_count'] + 1)

            # Match the syndrome interaction features used during training.
            # These fields are part of the fitted continuous schema in the bundled checkpoints.
            syndrome_encoded = df['Syndrome_encoded'] if 'Syndrome_encoded' in df.columns else 0
            df['syndrome_fever'] = syndrome_encoded * df.get('FEVER', 0)
            df['syndrome_respiratory'] = syndrome_encoded * \
                df['respiratory_symptoms']
            df['syndrome_gi'] = syndrome_encoded * df['gi_symptoms']
            df['syndrome_neuro'] = syndrome_encoded * df['neuro_symptoms']
            df['syndrome_skin'] = syndrome_encoded * df['skin_symptoms']
            df['syndrome_systemic'] = syndrome_encoded * df['systemic_symptoms']
            df['syndrome_severity'] = syndrome_encoded * df['severity_score']
            df['syndrome_age'] = syndrome_encoded * df['age']
            df['syndrome_symptom_count'] = syndrome_encoded * df['symptom_count']

            # Final cleanup
            df = df.replace([np.inf, -np.inf], 0).fillna(0)

            # ========== STANDARD PREPROCESSING ==========

            # === CONTINUOUS FEATURES ===
            # Ensure the dataframe contains all continuous columns the imputer/scaler
            # were fitted with. If missing, add them with default 0 values so
            # sklearn transformers won't raise feature-name mismatch errors.
            fitted_cont_cols = getattr(imp_cont, 'feature_names_in_', None)
            expected_cont_cols = list(
                fitted_cont_cols) if fitted_cont_cols is not None else list(cont_cols)
            for c in expected_cont_cols:
                if c not in df.columns:
                    df[c] = 0

            try:
                if expected_cont_cols and imp_cont is not None and scaler is not None:
                    # Use expected_cont_cols in the original fit order
                    X_cont = imp_cont.transform(df[expected_cont_cols])
                    X_cont = scaler.transform(X_cont).astype(np.float32)
                else:
                    X_cont = np.zeros(
                        (1, len(expected_cont_cols)), dtype=np.float32)
            except Exception as exc:
                # Defensive fallback: if transformer still errors, emit a warning
                # and use zeros to allow inference to continue.
                st.warning(
                    f"Continuous preprocessing failed; filling zeros. Error: {exc}")
                X_cont = np.zeros(
                    (1, len(expected_cont_cols)), dtype=np.float32)

            # === BINARY FEATURES ===
            available_bin_cols = [
                col for col in binary_cols if col in df.columns]
            if available_bin_cols:
                if imp_bin is not None:
                    X_bin = imp_bin.transform(
                        df[available_bin_cols]).astype(np.float32)
                else:
                    X_bin = df[available_bin_cols].fillna(
                        0).values.astype(np.float32)
            else:
                X_bin = np.zeros((1, len(binary_cols)), dtype=np.float32)

            # === CATEGORICAL FEATURES ===
            X_cat_list = []
            for col in cat_cols:
                if col in df.columns:
                    le = le_dict.get(col)
                    val = str(df[col].values[0])
                    if isinstance(le, LabelEncoder) or hasattr(le, 'classes_'):
                        mapping = {cls: idx for idx,
                                   cls in enumerate(le.classes_)}
                        encoded_val = mapping.get(val, 0)
                    else:
                        # Fallback: if this looks like a syndrome column, try CSV mapping or integer conversion
                        encoded_val = 0
                        try:
                            # direct integer present
                            raw_val = df[col].values[0]
                            if pd.notnull(raw_val):
                                try:
                                    encoded_val = int(raw_val)
                                except Exception:
                                    # try match label to SYNDROME_MAPPING
                                    lookup = str(raw_val).strip().lower()
                                    for k, v in SYNDROME_MAPPING.items():
                                        if str(v).strip().lower() == lookup:
                                            encoded_val = int(k)
                                            break
                        except Exception:
                            encoded_val = 0
                    X_cat_list.append(encoded_val)
                else:
                    X_cat_list.append(0)

            X_cat = np.array([X_cat_list], dtype=np.int64) if cat_cols else np.zeros(
                (1, 0), dtype=np.int64)

            # Convert to PyTorch tensors
            xb = torch.tensor(X_bin, dtype=torch.float32).to(DEVICE)
            xc = torch.tensor(X_cont, dtype=torch.float32).to(DEVICE)
            xcat = torch.tensor(X_cat, dtype=torch.long).to(DEVICE)

            return xb, xc, xcat

        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            import traceback
            st.error(traceback.format_exc())
            raise

    def predict(self, patient_data):
        """
        Complete prediction workflow: preprocess features and run both models.

        Args:
            patient_data: Dictionary with patient information

        Returns:
            dict: Prediction results with probabilities from both models
        """
        if self.model1 is None or self.model2 is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        try:
            # Drives the "should not appear" exclusion rules below.
            syndrome_encoded = patient_data.get(
                'Syndrome_encoded', patient_data.get('syndrome'))

            # Preprocess for Model 1
            xb1, xc1, xcat1 = self.preprocess_features(
                patient_data, self.preprocessing1)

            # Model 1 prediction (26 major viruses)
            with torch.no_grad():
                # Don't use return_embed for inference
                logits1 = self.model1(xb1, xc1, xcat1)
                y_pred_proba = torch.softmax(logits1, dim=1)[0].cpu().numpy()

            # Re-rank the top-5 excluding viruses that shouldn't appear for
            # this syndrome; the next-highest-confidence syndrome-consistent
            # prediction moves up to take the place of anything excluded.
            raw_top5 = np.argsort(y_pred_proba)[-5:][::-1]
            excluded_m1 = _build_excluded_index_set(
                syndrome_encoded, VIRUS_MAPPING)
            top_5_indices = _filter_topk(y_pred_proba, excluded_m1, k=5)
            y_pred = top_5_indices[0]
            excluded_from_view_m1 = [int(i)
                                     for i in raw_top5 if i in excluded_m1]

            # Check if "Other Viruses" (class 15) is in the (post-filter) top 5
            second_model_results = None
            if 15 in top_5_indices:
                xb2, xc2, xcat2 = self.preprocess_features(
                    patient_data, self.preprocessing2)

                with torch.no_grad():
                    # Don't use return_embed for inference
                    logits2 = self.model2(xb2, xc2, xcat2)
                    y_pred_proba_m2 = torch.softmax(logits2, dim=1)[
                        0].cpu().numpy()

                raw_top5_m2 = np.argsort(y_pred_proba_m2)[-5:][::-1]
                excluded_m2 = _build_excluded_index_set(
                    syndrome_encoded, OTHER_VIRUS_MAPPING)
                top_5_indices_m2 = _filter_topk(
                    y_pred_proba_m2, excluded_m2, k=5)
                y_pred_m2 = top_5_indices_m2[0]
                excluded_from_view_m2 = [
                    int(i) for i in raw_top5_m2 if i in excluded_m2]

                second_model_results = {
                    'prediction': y_pred_m2,
                    'probabilities': y_pred_proba_m2,
                    'top_5': top_5_indices_m2,
                    'excluded_by_syndrome': [OTHER_VIRUS_MAPPING[i] for i in excluded_from_view_m2],
                }

            return {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'top_5_indices': top_5_indices,
                'second_model_results': second_model_results,
                'excluded_by_syndrome': [VIRUS_MAPPING[i] for i in excluded_from_view_m1],
            }

        except Exception as e:
            st.error(f"Prediction error: {e}")
            raise


# ============================================================================
# STREAMLIT CACHE WRAPPER (for use in Streamlit apps)
# ============================================================================

@st.cache_resource
def get_virus_predictor():
    """
    Get or create a cached VirusPredictor instance (for Streamlit caching).

    Returns:
        VirusPredictor: Initialized predictor with loaded models
    """
    return VirusPredictor()
