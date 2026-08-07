"""
main.py

FastAPI backend that exposes our business logic (tools.py) as HTTP
endpoints.

Why do we need this layer if tools.py already has working functions?
-----------------------------------------------------------------------
Because the Realtime Voice Agent (running in the browser / OpenAI's
infrastructure) cannot directly import and call a Python function sitting
in our tools.py file. It needs something it can reach over the network.

So the flow becomes:

    Browser (WebRTC)
         |
    Realtime Agent (OpenAI)
         |  "I want to call check_coverage(...)"
         v
    Our FastAPI backend  <-- THIS FILE
         |
    tools.py functions
         |
    JSON mock data

Later, when we define "tool schemas" for the Realtime API, each tool will
essentially describe how to call one of the endpoints below.

Run this with:
    uvicorn main:app --reload
"""

import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our already-tested business logic functions
from tools import (
    get_patient,
    check_coverage,
    get_authorization_status,
    normalize_arabic_term,
)

# Load variables from a local .env file (e.g. OPENAI_API_KEY) into the
# environment. This means our real API key never has to be typed
# directly into any Python file — it stays in .env, which should NOT be
# committed to git.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="HealthVoice Backend",
    description="Mock healthcare insurance backend for the Voice AI demo",
)

# -----------------------------------------------------------------------
# CORS (Cross-Origin Resource Sharing)
# -----------------------------------------------------------------------
# Our frontend (a plain HTML/JS page) will be opened directly in the
# browser or served from a different local port than the backend
# (8000). Without CORS enabled, the browser will block the frontend's
# fetch() calls to this backend. For a local prototype, allowing all
# origins is fine — in production you would restrict this to your
# actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Request body models (Pydantic)
# -----------------------------------------------------------------------
# FastAPI uses these to validate incoming JSON automatically. If the
# request body is missing a field or has the wrong type, FastAPI will
# reject it with a clear error before our code even runs.

class CoverageRequest(BaseModel):
    insurance_provider: str  # e.g. "Bupa" (should be canonical, English)
    plan: str                # e.g. "Gold"
    procedure: str           # e.g. "Cardiac Surgery"


class AuthorizationRequest(BaseModel):
    patient_id: str          # e.g. "10203"
    procedure: str           # e.g. "Cardiac Surgery"


class NormalizeRequest(BaseModel):
    text: str                # raw phrase from the patient, Arabic or English
    category: str            # "procedures" or "insurance_providers"


# -----------------------------------------------------------------------
# 1) PATIENT LOOKUP
# -----------------------------------------------------------------------
# GET is fine here since we're just reading data using a simple ID in
# the URL path — no complex body needed.

@app.get("/api/patient/{patient_id}")
def get_patient_endpoint(patient_id: str):
    """
    Fetch a patient's record by ID.

    Example:
        GET /patient/10203

    Returns:
        200 + patient JSON if found
        404 if no patient exists with that ID
    """
    patient = get_patient(patient_id)

    if patient is None:
        # Raising HTTPException is how FastAPI returns proper HTTP error
        # codes (instead of just returning None, which would look like
        # a successful-but-empty response).
        raise HTTPException(
            status_code=404,
            detail=f"No patient found with ID '{patient_id}'",
        )

    return patient


# -----------------------------------------------------------------------
# 2) INSURANCE COVERAGE CHECK
# -----------------------------------------------------------------------
# POST here because we're sending multiple structured fields
# (provider + plan + procedure) — cleaner as a JSON body than cramming
# them into the URL as query params.

@app.post("/api/coverage")
def check_coverage_endpoint(request: CoverageRequest):
    """
    Check whether a procedure is covered under a given insurance
    provider + plan.

    IMPORTANT: insurance_provider, plan, and procedure must be canonical
    English values (e.g. "Bupa", "Gold", "Cardiac Surgery"). If the
    values came directly from Arabic speech, normalize them first via
    /normalize before calling this endpoint.

    Example request body:
        {
          "insurance_provider": "Bupa",
          "plan": "Gold",
          "procedure": "Cardiac Surgery"
        }

    Returns:
        200 + coverage details if a matching record exists
        404 if we have no coverage data for this exact combination
    """
    coverage = check_coverage(
        insurance_provider=request.insurance_provider,
        plan=request.plan,
        procedure=request.procedure,
    )

    if coverage is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No coverage record found for provider="
                f"'{request.insurance_provider}', plan='{request.plan}', "
                f"procedure='{request.procedure}'"
            ),
        )

    return coverage


# -----------------------------------------------------------------------
# 3) PRIOR AUTHORIZATION STATUS
# -----------------------------------------------------------------------

@app.post("/api/authorization")
def get_authorization_status_endpoint(request: AuthorizationRequest):
    """
    Check the prior-authorization request status for a patient +
    procedure (e.g. "has my approval come through yet?").

    Example request body:
        {
          "patient_id": "10203",
          "procedure": "Cardiac Surgery"
        }

    Returns:
        200 + authorization details if a record exists
        404 if no authorization request has been filed for this
        patient/procedure combination
    """
    authorization = get_authorization_status(
        patient_id=request.patient_id,
        procedure=request.procedure,
    )

    if authorization is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No authorization record found for patient_id="
                f"'{request.patient_id}', procedure='{request.procedure}'"
            ),
        )

    return authorization


# -----------------------------------------------------------------------
# 4) ARABIC TERM NORMALIZATION
# -----------------------------------------------------------------------
# Exposed as its own endpoint too, in case the agent (or our own testing)
# needs to normalize a term before calling /coverage or /authorization.

@app.post("/api/normalize")
def normalize_term_endpoint(request: NormalizeRequest):
    """
    Convert a raw Arabic/English phrase into its canonical English term.

    Example request body:
        {
          "text": "عملية قلب مفتوح",
          "category": "procedures"
        }

    Returns:
        200 + {"canonical_term": "Cardiac Surgery"} if a match is found
        404 if no match exists (caller should ask the patient to clarify)
    """
    canonical_term = normalize_arabic_term(
        text=request.text,
        category=request.category,
    )

    if canonical_term is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Could not match '{request.text}' to a known term "
                f"in category '{request.category}'"
            ),
        )

    return {"canonical_term": canonical_term}


# -----------------------------------------------------------------------
# 5) REALTIME SESSION TOKEN (ephemeral key)
# -----------------------------------------------------------------------
# WHY THIS ENDPOINT EXISTS:
# Our real OPENAI_API_KEY must NEVER be sent to or embedded in the
# browser — anyone could open Developer Tools and steal it. Instead, the
# browser asks OUR backend for a short-lived "ephemeral" token. Our
# backend (which safely holds the real API key, server-side only) asks
# OpenAI for that ephemeral token and hands it back to the browser.
# The browser then uses ONLY that short-lived token to open the WebRTC
# connection directly with OpenAI's Realtime API.
#
#     Browser --(1. GET /session)--> our FastAPI backend
#     our FastAPI backend --(2. uses real API key)--> OpenAI
#     OpenAI --(3. ephemeral token)--> our FastAPI backend
#     our FastAPI backend --(4. ephemeral token)--> Browser
#     Browser --(5. WebRTC using ephemeral token)--> OpenAI Realtime API
#
# We also attach our tool schemas + system instructions here, so every
# session the frontend starts already knows about get_patient,
# normalize_term, check_coverage, and get_authorization_status.

from tool_schemas import REALTIME_TOOLS

# Keep the system prompt here so it's easy to tweak without touching the
# rest of the endpoint logic.
SYSTEM_INSTRUCTIONS = """
You are a healthcare insurance voice assistant.

You speak Arabic and English fluently. Always respond in the same
language the user is speaking.

Keep responses concise and natural, suitable for being spoken aloud.

Never invent insurance coverage, authorization, or patient information.
For any factual question about a patient, coverage, or authorization
status, always use the available tools to get the real answer instead
of guessing.

If the patient mentions a procedure or insurance provider name in
Arabic or an informal phrase, call normalize_term first to get the
canonical English term before calling check_coverage or
get_authorization_status.

Never provide medical diagnosis or medical advice.

If you cannot confidently answer a question, or the patient asks to
speak to a human, let them know you will connect them with a human
agent.
""".strip()


@app.get("/api/session")
def create_realtime_session():
    """
    Request a short-lived ephemeral token from OpenAI's Realtime API,
    pre-configured with our tool schemas and system instructions.

    The frontend calls this endpoint FIRST (before opening WebRTC), and
    uses the returned ephemeral token/session data to establish the
    voice connection directly with OpenAI.

    Returns:
        The JSON session object from OpenAI (contains the ephemeral
        token under client_secret, plus session details).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is not set. Create a .env file in the "
                "backend folder (see .env.example) with your real key."
            ),
        )

    try:
        response = httpx.post(
            # NOTE: OpenAI's Realtime API moved from the old
            # /v1/realtime/sessions endpoint to /v1/realtime/client_secrets
            # in 2026. If you see 404s here, double check OpenAI's current
            # docs in case it moves again.
            "https://api.openai.com/v1/realtime/client_secrets",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                # The whole session config now lives nested under a
                # "session" key (this changed from the old flat format).
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "tools": REALTIME_TOOLS,
                    "audio": {
                        "output": {
                            "voice": "marin",
                        },
                    },
                },
            },
            timeout=15.0,
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach OpenAI: {exc}",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenAI session creation failed: {response.text}",
        )

    return response.json()


# -----------------------------------------------------------------------
# Health check endpoint
# -----------------------------------------------------------------------
# Simple endpoint to confirm the server is running. Useful when testing
# locally or when the frontend wants to check the backend is reachable
# before starting a voice session.

@app.get("/api/health")
def health_check():
    return {"status": "ok"}
