# Project review, September 10, 2026

**Share with caveats. Review coverage: partial.** The workflow is useful for conditional planning, but the evidence does not support calling it a calibrated lifetime financial forecast. This review repaired demonstrated errors, exercised the primary workflows, and refreshed the affected results. It did not certify every possible configuration or tax treatment.

## Material repairs

1. **Retirement accounting:** unpaid expenses and accrued taxes previously failed to reduce inferred saving capacity. A synthetic household with no income and $1.2 million of unpaid child expenses incorrectly qualified for retirement at 59.25. The corrected calculation does not qualify it. Tax accrual and immediate payment now imply the same resources. Results use `lifecycle_v3`; earlier retirement results are marked superseded.
2. **Restricted grants:** a milestone advance could release funds against another project's reimbursement claim. A $200 cost example invented $100 of unrestricted cash. Advances now remain project-specific; tests cover both overlapping projects and insufficient working capital.
3. **Configuration and coverage:** misspelled base-schema fields and repeated grid values could silently change choices or duplicate scenario IDs. Both are rejected before enumeration. The salary analysis respects the configured minimum sample size.
4. **Recovery and integrity:** resume can retry interrupted reports without repeating completed simulations, and completed resumes verify partition hashes. Source archives include the missing dashboard dependency. `validate_run.py --output <folder> --deep` independently checks committed hashes, counts, duplicate identifiers, summary medians and ordered monthly bands without modifying results.
5. **Usability and interpretation:** reset filters, visible catalog scope, salary CSV exports with requirement/age context, explicit independent filter scope, and readable invalid-folder errors. README setup starts with the 60-scenario demo and preserves an existing private configuration. The selected gallery is no longer mislabeled as the entire standard preset.

## Dashboard best practices and quality

Counts refer to the scoped inventory, not proof of exhaustive verification. Zero observed defects means no remaining demonstrated defect in that category. Repeated states and charts were sampled; narrow-screen rendering was not verified in this review.

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Usefulness and completeness | 0 / 4 | Overview, observed trends, controlled salary analysis and run workflow answer distinct questions. Full lifetime tax/career simulation and calibrated regional inputs remain outside the completed scope. |
| Analytical clarity | 0 / 4 | Scope, uncertainty, superseded results and long-horizon assumptions are visible. “All choices” refers to tested configurations and paired economic draws. |
| Visual and interaction consistency | 0 / 5 | Representative filtering/reset, signed axes, requirement buttons, age selection and invalid/partial states checked. No exhaustive combination or narrow-screen certification. |

## Analytical correctness and robustness

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Source authority and confidence | 0 / 3 | Federal bracket anchors, Social Security wage base and college-cost components checked against primary sources. This is a targeted source check, not validation of every input. |
| Calculation accuracy | 1 / 6 | Household reconciliation, grant restrictions, salary inference, tax anchors, lifetime expenses and retirement inspected with tests. Negative accessible capital still receives a portfolio multiplier after 2036; see the remaining problem below. |
| Within-chart agreement | 0 / 4 | Overview, risk trends, salary success and salary breakdown share checked values and scope labels. |
| Complete source details | 0 / 3 | Saved configuration, dated source records and numerical-engine archives retained. Historical archives remain immutable, including their original omissions. |
| Cross-artifact consistency | 0 / 3 | Main results/gallery, retirement version labels, and salary criteria/exports checked. Main catalog and salary catalog intentionally have different coverage. |
| Data-quality controls | 0 / 4 | Configuration validation, committed partitions, independent-path counting and partial-run handling checked. |
| Conclusion support | 1 / 3 | Liquidity trends and salary sampling statements remain conditional. Earliest retirement ages on projected deficit paths need the stronger lifetime accounting described below. |

## Prioritized remaining problems and proposed fixes

1. **Projected deficits after 2036.** Negative accessible capital still receives the accumulation return multiplier. That is not an explicit borrowing, repayment or insolvency model. Proposed fix: separate investable assets, outstanding liabilities and unmet obligations throughout the lifetime extension, and test deficit recovery without assuming borrowing capacity. Until then, treat earliest retirement ages on such paths as provisional. This limitation is separate from the repaired phantom-income error.
2. **Long-horizon model risk.** Income capacity is inferred from recent years; the ten-year macro narrative does not continue as a full lifetime career, tax, Social Security, pension or healthcare simulation. Proposed fix: a dated lifetime cash-flow engine with retirement withdrawal taxes and benefit eligibility, plus sensitivity comparisons to the current conservative finite-obligation reserve.
3. **Uncalibrated costs and legal treatments.** Local housing/childcare, Amanda's benefits and several tax treatments are assumptions. State forms, AMT, QBI and other exclusions are documented in [MODEL.md](MODEL.md). Proposed fix: replace the consequential assumptions with geographic/provider evidence and validate representative returns against an independent tax calculator before treating salary thresholds as financial advice.
4. **Sampling and coverage.** At 256 independent paths, a point estimate can clear 95% while the simultaneous lower bound does not. More precision requires additional independent paths, not more correlated scenario rows. The refreshed main catalog contains 5,742 selected scenarios; each salary catalog tests 1,548 configured life choices at 13 salary points. Neither covers every annual birth schedule. The full configurable grid remains available and its runtime is explicit.

## Source checks

The implemented single/joint standard deductions and marginal brackets match the [IRS 2026 inflation adjustments](https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill). The payroll wage-base anchor matches the [SSA contribution and benefit base](https://www.ssa.gov/OACT/COLA/cbb.html). These checks do not establish complete tax-return accuracy.

The college allowance uses the [NCES public four-year institutional cost components](https://nces.ed.gov/ipeds/search/viewtable?returnUrl=/search&tableId=36306), with an explicitly assumed inflation projection and the user's 50% contribution. It is not a school quote or enrollment-weighted household cost estimate.

## Completed verification

- Full suite: 103 tests passed; the final reset-persistence change also passed the 12 affected app/salary tests.
- All 11,773,440 requested paths rerun across the same main and two salary catalogs. Every committed partition passed integrity checks.
- Main catalog: 5,742 scenarios with identical ten-year wealth and distress outcomes; 321 median retirement ages changed.
- Salary probabilities and simultaneous lower bounds independently reconciled from retained success flags at both age targets.
- Workbook opened and all eight sheets checked. All 27 PNGs are 300 dpi; only the retirement figure changed from the previously inspected set and was inspected again.
- Browser checks covered empty selections, reset and subsequent rerun, source/coverage labels, and age-65 uncertainty. Invisible configuration-range marks were replaced with visible triangles; the salary-status column was widened and visually rechecked to eliminate clipping.

## Reproduce checks

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
python validate_run.py --output outputs/decision-audited-2026-09-10 --deep
python validate_run.py --output outputs/salary-support-audited-zero-2026-09-10 --deep
python validate_run.py --output outputs/salary-support-audited-darpa-2026-09-10 --deep
```

Private audit evidence and before/after comparisons remain in `outputs/review-salary-support/`. Published gallery assets contain figures and aggregate metadata only. No commit or publication was performed.
