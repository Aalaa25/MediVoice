"""
tools.py

This file contains the "raw" helper functions that will later be wrapped
as Tools for the Realtime Voice Agent.

Why separate "helper functions" from "tools"?
-----------------------------------------------
Right now these are just plain Python functions that read from local JSON
files (our mock database). Later, we will:
  1. Expose these as FastAPI endpoints (backend/main.py)
  2. Define a "tool schema" (JSON) for each one, so the LLM/Realtime Agent
     knows it exists and what parameters it needs
  3. When the agent calls a tool, our code will call these same functions
     (or hit the FastAPI endpoint) and return the result back to the agent

Keeping the logic here, decoupled from the AI/voice layer, means:
  - We can unit test this logic without needing OpenAI's Realtime API
  - We can swap the mock JSON files for a real database/API later without
    touching the AI layer at all
"""

import json
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------
# All our mock data lives in backend/data/*.json
DATA_DIR = Path(__file__).parent / "data"


def _load_json(filename: str):
    """
    Small internal utility to load a JSON file from the data/ folder.
    Not meant to be called directly from outside this module (hence the
    leading underscore) — it's just here to avoid repeating
    "open + json.load" in every function below.
    """
    file_path = DATA_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------
# 1) ARABIC TERM NORMALIZATION
# -----------------------------------------------------------------------
# Patients will say things in different ways:
#   "عملية قلب مفتوح" / "عملية القلب" / "جراحة قلب" -> all mean "Cardiac Surgery"
#   "بوبا" -> "Bupa"
# Before we call check_coverage() or get_authorization_status(), we need
# to convert whatever the patient said into the "canonical" English term
# that matches the keys used in our other JSON files (insurance_coverage.json,
# authorization_status.json, etc).
#
# This function does NOT use the LLM to do this — it's a straightforward
# dictionary/lookup match against arabic_terms.json. Using a real
# code-based lookup here (instead of letting the LLM "guess" the English
# term) is important for accuracy in a healthcare context.

def normalize_arabic_term(text: str, category: str) -> Optional[str]:
    """
    Convert a raw user phrase (Arabic or English, exact or a known variant)
    into its canonical English term.

    Args:
        text: The raw phrase the patient said, e.g. "عملية قلب مفتوح" or "Bupa".
        category: Which lookup table to search in arabic_terms.json.
                  Must be one of: "procedures", "insurance_providers".

    Returns:
        The canonical English term (e.g. "Cardiac Surgery", "Bupa") if a
        match is found, otherwise None.

    Example:
        normalize_arabic_term("عملية قلب مفتوح", "procedures")
        -> "Cardiac Surgery"

        normalize_arabic_term("بوبا", "insurance_providers")
        -> "Bupa"
    """
    terms_data = _load_json("arabic_terms.json")

    # Guard: make sure the requested category actually exists in the file
    if category not in terms_data:
        return None

    # Normalize the input text a bit (strip whitespace) so small typing
    # differences like extra spaces don't break the match
    cleaned_text = text.strip()

    for entry in terms_data[category]:
        # Direct match against the canonical English term itself
        # (case-insensitive, so "cardiac surgery" also matches)
        if cleaned_text.lower() == entry["canonical"].lower():
            return entry["canonical"]

        # Direct match against the main Arabic term
        if cleaned_text == entry["ar"]:
            return entry["canonical"]

        # Match against any of the known Arabic/English variants/synonyms
        if cleaned_text in entry.get("variants", []):
            return entry["canonical"]

    # No match found — the caller should handle this by asking the
    # patient to clarify (e.g. "Sorry, could you repeat the procedure name?")
    return None


def detect_intent(text: str) -> Optional[str]:
    """
    Very lightweight intent matcher based on arabic_terms.json's
    "common_intents" section. This is NOT meant to replace the LLM's own
    understanding — the Realtime Agent will mostly decide intent on its
    own using the conversation. This helper is more useful for:
      - Logging / debugging (to see what intent was detected)
      - Fallback matching if we ever run this outside the LLM

    Args:
        text: Raw user utterance, e.g. "هل التأمين يغطي العملية؟"

    Returns:
        One of "check_coverage", "check_authorization",
        "authorization_status", or None if nothing matches.
    """
    terms_data = _load_json("arabic_terms.json")
    cleaned_text = text.strip()

    for intent_entry in terms_data.get("common_intents", []):
        for example_phrase in intent_entry.get("examples", []):
            # Simple substring check — good enough for a demo/prototype.
            # A production system would use embeddings or the LLM itself.
            if example_phrase in cleaned_text or cleaned_text in example_phrase:
                return intent_entry["intent"]

    return None


# -----------------------------------------------------------------------
# 2) PATIENT LOOKUP
# -----------------------------------------------------------------------

def get_patient(patient_id: str) -> Optional[dict]:
    """
    Look up a patient's record by their patient ID.

    Args:
        patient_id: The patient's ID as a string, e.g. "10203".
                    (Kept as string since IDs come from voice/text input
                    and we don't want int-parsing errors to crash things.)

    Returns:
        A dict with the patient's info (name, insurance_provider, plan,
        preferred_language, etc.) if found, otherwise None.

    Example:
        get_patient("10203")
        -> {
             "patient_id": "10203",
             "name": "Sara Ahmed",
             "insurance_provider": "Bupa",
             "plan": "Gold",
             ...
           }
    """
    patients = _load_json("patients.json")

    for patient in patients:
        if patient["patient_id"] == str(patient_id).strip():
            return patient

    # No patient found with this ID
    return None


# -----------------------------------------------------------------------
# 3) INSURANCE COVERAGE LOOKUP
# -----------------------------------------------------------------------

def check_coverage(
    insurance_provider: str,
    plan: str,
    procedure: str,
) -> Optional[dict]:
    """
    Check whether a given procedure is covered under a specific insurance
    provider + plan combination.

    IMPORTANT: The caller is expected to pass CANONICAL English values here
    (e.g. "Bupa", "Gold", "Cardiac Surgery") — not raw Arabic phrases.
    Run normalize_arabic_term() first if the input came directly from the
    patient's speech.

    Args:
        insurance_provider: Canonical provider name, e.g. "Bupa".
        plan: Canonical plan name, e.g. "Gold".
        procedure: Canonical procedure name, e.g. "Cardiac Surgery".

    Returns:
        A dict with coverage details (covered, coverage_percentage,
        patient_copay_percentage, prior_authorization_required,
        network_required, notes) if a matching record is found,
        otherwise None (meaning we have no data for this combination —
        the agent should say it can't confirm and offer human handoff).

    Example:
        check_coverage("Bupa", "Gold", "Cardiac Surgery")
        -> {
             "covered": true,
             "coverage_percentage": 80,
             "patient_copay_percentage": 20,
             "prior_authorization_required": true,
             "network_required": true,
             "notes": "Procedure must be performed at an approved
                        in-network hospital."
           }
    """
    coverage_records = _load_json("insurance_coverage.json")

    for record in coverage_records:
        matches_provider = record["insurance_provider"].lower() == insurance_provider.strip().lower()
        matches_plan = record["plan"].lower() == plan.strip().lower()
        matches_procedure = record["procedure"].lower() == procedure.strip().lower()

        if matches_provider and matches_plan and matches_procedure:
            return record

    # No coverage record exists for this exact combination
    return None


# -----------------------------------------------------------------------
# 4) PRIOR AUTHORIZATION STATUS LOOKUP
# -----------------------------------------------------------------------

def get_authorization_status(patient_id: str, procedure: str) -> Optional[dict]:
    """
    Look up the prior-authorization request status for a specific patient
    and procedure (e.g. "has the approval come through yet?").

    Args:
        patient_id: The patient's ID, e.g. "10203".
        procedure: Canonical procedure name, e.g. "Cardiac Surgery".

    Returns:
        A dict with authorization details (status, submitted_at, decision)
        if a matching record exists, otherwise None (meaning no
        authorization request has been filed for this patient/procedure
        yet — different from "required_not_submitted", which means we
        know one is needed but it just hasn't been sent).

    Example:
        get_authorization_status("10203", "Cardiac Surgery")
        -> {
             "authorization_id": "AUTH-1001",
             "status": "required_not_submitted",
             "submitted_at": None,
             "decision": None
           }
    """
    auth_records = _load_json("authorization_status.json")

    for record in auth_records:
        matches_patient = record["patient_id"] == str(patient_id).strip()
        matches_procedure = record["procedure"].lower() == procedure.strip().lower()

        if matches_patient and matches_procedure:
            return record

    # No authorization record found for this patient + procedure combo
    return None


# -----------------------------------------------------------------------
# Quick manual test (only runs if you execute this file directly:
# `python tools.py`). Not part of the actual app — just a sanity check
# while building, so you can confirm the JSON files are wired correctly.
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing normalize_arabic_term:")
    print(normalize_arabic_term("عملية قلب مفتوح", "procedures"))  # -> Cardiac Surgery
    print(normalize_arabic_term("بوبا", "insurance_providers"))     # -> Bupa

    print("\nTesting get_patient:")
    print(get_patient("10203"))

    print("\nTesting check_coverage:")
    print(check_coverage("Bupa", "Gold", "Cardiac Surgery"))

    print("\nTesting get_authorization_status:")
    print(get_authorization_status("10203", "Cardiac Surgery"))
