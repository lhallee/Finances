"""Presentation labels; serialized configuration and scenario IDs stay unchanged."""

import pandas as pd


NAMES = {
    "bio_ai": "Bio-AI / Protein Design", "all_tested_macros": "All Tested Economic Scenarios", "all_tested_locations": "All Tested Locations",
    "ud": "University of Delaware", "ud_blend": "UD + Synthyra", "ud_nonrenewal": "UD Nonrenewal",
    "biohub_engineer": "Biohub Engineer", "biohub_scientist": "Biohub Scientist",
    "profluent": "Profluent", "merck": "Pharma / Agritech Leadership (Legacy Pay Case)", "synthyra": "Synthyra",
    "pharma_agritech_pi": "Pharma / Agritech Principal Scientist",
    "pharma_agritech_director": "Pharma / Agritech Research Director",
    "bio_ai_engineer": "Bio-AI Engineering", "bio_ai_research": "Bio-AI Research",
    "protein_design": "Protein Design Research",
    "faculty_ud": "UD Faculty", "faculty_ca": "California Faculty", "faculty_ma": "Massachusetts Faculty",
    "industry_blend": "Industry + Synthyra", "faculty_blend": "Faculty + Synthyra",
    "parents": "Living With Parents", "squirrel_hill": "Squirrel Hill, Pittsburgh",
    "bethel": "Bethel, Maine", "norway": "Norway / South Paris", "farmington": "Farmington, Maine",
    "newark": "Newark, Delaware", "philadelphia": "Philadelphia Suburbs", "nyc": "New York City",
    "bay_area": "Bay Area", "san_diego": "San Diego", "boston": "Boston / Cambridge", "amherst": "Western Massachusetts",
    "saas": "SaaS", "vc": "Venture Capital", "darpa": "DARPA SBIR", "sbir_phase1": "SBIR Phase I",
    "ud_contract": "UD Contract", "pharma_agritech": "Pharma / Agritech Research",
    "ai_modest": "Modest AI Transition", "ai_substantial": "Substantial AI Transition", "ai_extreme": "Extreme AI Transition",
    "wet_lab": "Wet Lab", "animal_science": "Animal Science", "data_science": "Data Science",
    "free": "Rent-Free", "buy": "Buy When Affordable", "rent": "Rent",
    "direct": "Specified Outcomes", "business": "Business-Driven", "equity": "Equity Sale", "asset": "Asset Sale",
}


def career_group(value: str) -> str:
    """One visual option per career family, preserving the saved underlying role."""
    if value in ("biohub_engineer", "biohub_scientist", "profluent", "bio_ai_engineer", "bio_ai_research", "protein_design"):
        return "Bio-AI / Protein Design"
    if value in ("merck", "pharma_agritech_pi", "pharma_agritech_director"):
        return "Pharma / Agritech PI & Director"
    return display_name(value)


def display_name(value: object) -> str:
    text = str(value)
    return NAMES.get(text, text.replace("_", " ").title())


def option_name(field: str, value: object) -> str:
    if field == "stop_after_birth":
        return "Continue Working" if int(value) == 0 else f"Stop After Child {int(value)}"
    if field in ("marriage_year", "exit_year"):
        return ("Remain Unmarried" if field == "marriage_year" else "No Exit") if int(value) == 0 else str(int(value))
    if field == "exit_value":
        return "No Exit" if float(value) == 0 else f"${float(value)/1e6:,.0f} Million"
    if field == "births":
        return str(value) or "No Children"
    if field == "grants":
        return " + ".join(display_name(item) for item in str(value).split(",") if item) or "No Grants"
    if isinstance(value, str):
        return display_name(value)
    return str(value)


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    categories = ("career", "location", "housing", "company_outcome", "company_mode", "macro", "benefits",
                  "amanda_career", "salary_family", "exit_type", "grant_case", "grants", "births", "stop_after_birth", "marriage_year")
    for field in categories:
        if field in formatted:
            formatted[field] = formatted[field].map(lambda value: option_name(field, value))
    return formatted.rename(columns=lambda column: str(column).replace("_", " ").title().replace("Id", "ID").replace("Mc ", "MC "))
