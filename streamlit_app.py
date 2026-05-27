"""
SCA Sustainability Project Idea Generator
==========================================

A Streamlit app that helps Temasek Polytechnic students generate
sustainability-related SCA group project ideas using Google's Gemini model.

The prompt template and reference context are loaded from markdown files
in the /prompts directory, so non-technical staff can update them by
editing those files directly (no Python knowledge required).
"""

import os
import re
from pathlib import Path

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("SCAle")
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).parent / "prompts"

PROMPT_FILE = PROMPTS_DIR / "system_prompt.md"
DIPLOMA_FILE = PROMPTS_DIR / "diploma_descriptions.md"
GUIDELINES_FILE = PROMPTS_DIR / "project_guidelines.md"

# Placeholders that must exist in the system prompt.
# If any of these are missing, the app will warn at startup.
REQUIRED_PLACEHOLDERS = [
    "{{diploma}}",
    "{{concern}}",
    "{{focus_area}}",
    "{{theme}}",
    "{{presentation}}",
    "{{extra_input}}",
    "{{project_guidelines}}",
    "{{diploma_description}}",
]

# The 12 climate change challenge topics from the SCA Group Project Brief,
# aligned with Singapore's Green Plan 2030.
GREEN_PLAN_FOCUS_AREAS = [
    "Circular Economy",
    "Liveable City and Community",
    "Green Buildings",
    "Renewable Energy",
    "Green Finance and Impact Investment",
    "Sustainable Food System / Food Security",
    "Sustainable Materials / Packaging",
    "Green Transportation",
    "Sustainable / Regenerative Tourism",
    "Green Economy Opportunities",
    "Waste Management & Recycling",
    "Biodiversity and Conservation",
]

# Deliverable formats from the project brief.
PRESENTATION_MODES = [
    "Physical prototype or model",
    "Digital prototype (app/website mock-up, model drawings)",
    "Social media campaign (3 sample posts)",
    "Short video (~1 minute)",
    "Podcast (~3 minutes)",
]


# ---------------------------------------------------------------------------
# File loading (cached so we only read from disk once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_text_file(path: Path) -> str:
    """Read a markdown/text file from disk. Cached for the session."""
    return path.read_text(encoding="utf-8")


def extract_diploma_names(diploma_markdown: str) -> list[str]:
    """
    Pull diploma names out of the diploma_descriptions.md file by looking
    for level-2 headings (## Diploma in ...).

    This way, when colleagues add a new diploma to the markdown file,
    it automatically appears in the dropdown.
    """
    names = re.findall(r"^##\s+(Diploma\s+in\s+.+?)\s*$", diploma_markdown, re.MULTILINE)
    return names


def extract_diploma_section(diploma_markdown: str, diploma_name: str) -> str:
    """
    Extract just the section for the selected diploma from the full
    diploma descriptions file. Keeps the prompt focused.
    """
    pattern = rf"##\s+{re.escape(diploma_name)}\s*\n(.*?)(?=\n##\s+|\Z)"
    match = re.search(pattern, diploma_markdown, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the whole file if we can't isolate the section.
    return diploma_markdown


def validate_prompt_placeholders(prompt_text: str) -> list[str]:
    """Return a list of required placeholders that are missing from the prompt."""
    return [p for p in REQUIRED_PLACEHOLDERS if p not in prompt_text]


def fill_prompt(template: str, values: dict) -> str:
    """
    Replace {{placeholder}} tokens in the template with actual values.

    Using simple string replacement (not Python's .format() or f-strings)
    so that any stray curly braces in the markdown won't cause errors.
    """
    filled = template
    for key, value in values.items():
        filled = filled.replace(f"{{{{{key}}}}}", str(value))
    return filled


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

@st.cache_resource
def get_llm():
    """
    Initialise the Gemini LLM. Reads the API key from Streamlit secrets
    (.streamlit/secrets.toml) or from the GOOGLE_API_KEY environment variable.
    """
    api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "No Google API key found. Please add `GOOGLE_API_KEY` to "
            "`.streamlit/secrets.toml` or set it as an environment variable."
        )
        st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,  # a touch of creativity for varied ideas
        google_api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SCA Sustainability Project Idea Generator",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 SCA Sustainability Project Idea Generator")
st.caption(
    "Get tailored project ideas aligned with your diploma and Singapore's "
    "Green Plan 2030."
)

# ---- Load prompt files & validate -----------------------------------------

try:
    prompt_template = load_text_file(PROMPT_FILE)
    diploma_markdown = load_text_file(DIPLOMA_FILE)
    guidelines_markdown = load_text_file(GUIDELINES_FILE)
except FileNotFoundError as e:
    st.error(
        f"Could not find a required prompt file: `{e.filename}`. "
        "Please make sure the `prompts/` folder exists with all three files."
    )
    st.stop()

missing = validate_prompt_placeholders(prompt_template)
if missing:
    st.error(
        "The system prompt is missing required placeholders: "
        + ", ".join(f"`{m}`" for m in missing)
        + ". Please check `prompts/system_prompt.md` and add them back."
    )
    st.stop()

diploma_names = extract_diploma_names(diploma_markdown)
if not diploma_names:
    st.error(
        "No diplomas could be found in `prompts/diploma_descriptions.md`. "
        "Each diploma should be under a `## Diploma in ...` heading."
    )
    st.stop()

# ---- Form -----------------------------------------------------------------

st.markdown("### Tell us about your project")

with st.form("project_form"):
    diploma = st.selectbox(
        "Diploma Programme",
        options=diploma_names,
        help="Select your diploma so suggestions match your skills.",
    )

    concern = st.text_input(
        "Sustainability Concern",
        placeholder="e.g. Food waste in hawker centres",
        help="What sustainability issue interests or worries you?",
    )

    focus_area = st.selectbox(
        "Green Plan Focus Area",
        options=GREEN_PLAN_FOCUS_AREAS,
        help="Choose one of the climate change challenge topics from the SCA Group Project brief.",
    )

    theme = st.text_input(
        "Project Theme",
        placeholder="e.g. Awareness campaign for students",
        help="A specific angle or theme you'd like to explore.",
    )

    presentation = st.multiselect(
        "Preferred Deliverable Format(s)",
        options=PRESENTATION_MODES,
        help="Select one or more formats you'd be interested in producing.",
    )

    extra_input = st.text_area(
        "Additional Preferences (optional)",
        placeholder="Anything else? Budget constraints, group size, technical skills, target audience...",
        height=100,
    )

    submitted = st.form_submit_button("✨ Generate Project Ideas", type="primary")

# ---- Optional: prompt preview (helpful for staff verifying edits) ----------

with st.expander("🔍 Preview the assembled prompt (for staff)"):
    st.caption(
        "This shows exactly what gets sent to the AI based on your form input. "
        "Useful for verifying that edits to the markdown files are working as expected."
    )
    if diploma and concern and focus_area and theme:
        preview = fill_prompt(
            prompt_template,
            {
                "diploma": diploma,
                "concern": concern,
                "focus_area": focus_area,
                "theme": theme,
                "presentation": ", ".join(presentation) if presentation else "(not specified)",
                "extra_input": extra_input or "None",
                "project_guidelines": guidelines_markdown,
                "diploma_description": extract_diploma_section(diploma_markdown, diploma),
            },
        )
        st.code(preview, language="markdown")
    else:
        st.info("Fill in the form fields above to see the assembled prompt.")

# ---- Handle submission ----------------------------------------------------

if submitted:
    # Validate required fields
    if not all([diploma, concern, focus_area, theme]) or not presentation:
        st.warning("Please complete all required fields before generating ideas.")
        st.stop()

    # Build the final prompt
    final_prompt = fill_prompt(
        prompt_template,
        {
            "diploma": diploma,
            "concern": concern,
            "focus_area": focus_area,
            "theme": theme,
            "presentation": ", ".join(presentation),
            "extra_input": extra_input.strip() if extra_input.strip() else "None",
            "project_guidelines": guidelines_markdown,
            "diploma_description": extract_diploma_section(diploma_markdown, diploma),
        },
    )

    # Call the LLM
    llm = get_llm()
    with st.spinner("Generating ideas tailored to your profile..."):
        try:
            response = llm.invoke(final_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            st.error(f"Something went wrong while generating ideas: {e}")
            st.stop()

    st.markdown("### 💡 Your Project Ideas")
    st.markdown(answer)

    st.download_button(
        label="📥 Download ideas as Markdown",
        data=answer,
        file_name=f"sca_project_ideas_{diploma.replace(' ', '_')}.md",
        mime="text/markdown",
    )

# ---- Footer ---------------------------------------------------------------

st.divider()
st.caption(
    "These suggestions are AI-generated starting points. Always discuss with "
    "your tutor and refer to the official SCA Group Project Brief for the "
    "final assessment criteria."
)
