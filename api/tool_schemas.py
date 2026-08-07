"""
tool_schemas.py

This file defines the "tools" (functions) that we tell the OpenAI
Realtime API about, so the voice agent knows:
  1. That these functions exist
  2. What each one does (so it decides WHEN to call it)
  3. What parameters each one needs (so it extracts the right values
     from the conversation before calling it)

IMPORTANT: This is just the DESCRIPTION of the tools (a JSON schema).
It does NOT contain the actual logic — the real logic lives in
tools.py / main.py. When the Realtime API decides to call a tool, it
sends back a "function_call" event with the tool name + arguments.
Our own code (the WebRTC/session handler we build next) is responsible
for:
    1. Receiving that function_call event
    2. Calling the matching FastAPI endpoint (or tools.py function
       directly) with those arguments
    3. Sending the result back to the Realtime API as a
       "function_call_output" event, so the agent can keep talking

Flow recap:
    Patient speech
        -> Realtime API (understands intent + extracts parameters)
        -> "I want to call check_coverage(...)"   <-- uses THIS schema
        -> our code executes it against FastAPI/tools.py
        -> result sent back to Realtime API
        -> Realtime API turns the result into a natural language answer
        -> TTS speaks it back to the patient

How to use this file:
    These schemas get passed into the Realtime session config, e.g.:

        session_config = {
            "instructions": "...",
            "tools": REALTIME_TOOLS,
            ...
        }

    (Exact key names can differ slightly depending on which Realtime API
    version/SDK you're using — check the Playground's generated code
    once you set tools up there, and adjust key names if needed.)
"""

# -----------------------------------------------------------------------
# Tool 1: get_patient
# -----------------------------------------------------------------------
# When should the agent call this?
#   As soon as the patient gives their patient ID, so we know who we're
#   talking to (name, insurance provider, plan). This is usually the
#   FIRST tool called in a conversation.

get_patient_tool = {
    "type": "function",
    "name": "get_patient",
    "description": (
        "Look up a patient's record using their patient ID. Returns the "
        "patient's name, insurance provider, plan, and preferred "
        "language. Call this as soon as the patient provides their "
        "patient ID number. If the patient ID is not found, inform the "
        "patient politely and ask them to double check the number."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": (
                    "The patient's ID number, exactly as given by the "
                    "patient, e.g. '10203'."
                ),
            },
        },
        "required": ["patient_id"],
    },
}


# -----------------------------------------------------------------------
# Tool 2: normalize_term
# -----------------------------------------------------------------------
# When should the agent call this?
#   Whenever the patient mentions a procedure name or insurance provider
#   name in Arabic (or an informal/variant phrase), BEFORE calling
#   check_coverage or get_authorization_status. This converts things
#   like "عملية قلب مفتوح" into the canonical "Cardiac Surgery" that our
#   other tools expect.

normalize_term_tool = {
    "type": "function",
    "name": "normalize_term",
    "description": (
        "Convert a raw phrase the patient said (in Arabic or English, "
        "including informal variants) into its canonical English term. "
        "ALWAYS call this before calling check_coverage or "
        "get_authorization_status if the procedure or insurance "
        "provider name came directly from the patient's speech and you "
        "are not already certain of its canonical English form. "
        "If no match is found, ask the patient to clarify or repeat the "
        "procedure/provider name rather than guessing."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "The raw phrase as the patient said it, e.g. "
                    "'عملية قلب مفتوح' or 'بوبا'."
                ),
            },
            "category": {
                "type": "string",
                "enum": ["procedures", "insurance_providers"],
                "description": (
                    "Which category to search in. Use 'procedures' for "
                    "medical procedure names, or 'insurance_providers' "
                    "for insurance company names."
                ),
            },
        },
        "required": ["text", "category"],
    },
}


# -----------------------------------------------------------------------
# Tool 3: check_coverage
# -----------------------------------------------------------------------
# When should the agent call this?
#   When the patient asks whether a procedure is covered, or how much of
#   it is covered. Requires canonical (English) values for provider,
#   plan, and procedure — so normalize_term should typically be called
#   first if these came from raw speech.

check_coverage_tool = {
    "type": "function",
    "name": "check_coverage",
    "description": (
        "Check whether a specific medical procedure is covered under a "
        "given insurance provider and plan. Returns whether it's "
        "covered, the coverage percentage, the patient's copay "
        "percentage, whether prior authorization is required, and any "
        "notes. Use the canonical English values for insurance_provider, "
        "plan, and procedure (call normalize_term first if needed). "
        "Never guess or make up coverage information — always call this "
        "tool to get the real answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "insurance_provider": {
                "type": "string",
                "description": (
                    "Canonical insurance provider name, e.g. 'Bupa', "
                    "'AXA', or 'Cigna'."
                ),
            },
            "plan": {
                "type": "string",
                "description": (
                    "Canonical plan name, e.g. 'Gold', 'Silver', "
                    "'Premium', or 'Basic'."
                ),
            },
            "procedure": {
                "type": "string",
                "description": (
                    "Canonical procedure name, e.g. 'Cardiac Surgery', "
                    "'MRI', or 'Knee Replacement'."
                ),
            },
        },
        "required": ["insurance_provider", "plan", "procedure"],
    },
}


# -----------------------------------------------------------------------
# Tool 4: get_authorization_status
# -----------------------------------------------------------------------
# When should the agent call this?
#   When the patient asks about the status of a prior-authorization
#   request (e.g. "did my approval come through?").

get_authorization_status_tool = {
    "type": "function",
    "name": "get_authorization_status",
    "description": (
        "Check the prior-authorization request status for a patient and "
        "procedure. Returns the status (e.g. approved, pending, or "
        "required but not yet submitted), when it was submitted, and "
        "the decision if one has been made. Use the canonical English "
        "procedure name (call normalize_term first if needed)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "patient_id": {
                "type": "string",
                "description": "The patient's ID number, e.g. '10203'.",
            },
            "procedure": {
                "type": "string",
                "description": (
                    "Canonical procedure name, e.g. 'Cardiac Surgery'."
                ),
            },
        },
        "required": ["patient_id", "procedure"],
    },
}


# -----------------------------------------------------------------------
# Combined list — this is what gets passed into the Realtime session
# config's "tools" field.
# -----------------------------------------------------------------------

REALTIME_TOOLS = [
    get_patient_tool,
    normalize_term_tool,
    check_coverage_tool,
    get_authorization_status_tool,
]


# -----------------------------------------------------------------------
# Quick manual check: print the schemas as pretty JSON, so you can copy
# them into the Realtime Playground's "tools" config field if needed.
# Run with: python tool_schemas.py
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print(json.dumps(REALTIME_TOOLS, indent=2, ensure_ascii=False))
