"""Common, reproducible 2026-dollar starting compensation draws by career family."""

import numpy as np

from .configuration import RunConfig
from .reference import CAREER_SALARY


BIO_AI = ("bio_ai_engineer", "bio_ai_research", "protein_design")
PHARMA = ("pharma_agritech_pi", "pharma_agritech_director")


def starting_salaries(config: RunConfig, career: str) -> np.ndarray:
    if career in config.career_salary_overrides:
        return np.full(config.paths, config.career_salary_overrides[career])  # (paths,)
    family = "bio_ai" if career in BIO_AI else "pharma_agritech" if career in PHARMA else None
    if family is None:
        return np.full(config.paths, CAREER_SALARY[career])  # (paths,)
    limits = config.career_salary_ranges[family]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1701]))
    return rng.triangular(limits.minimum, limits.mode, limits.maximum, size=config.paths)  # (paths,)
