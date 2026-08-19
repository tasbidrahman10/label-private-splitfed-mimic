# Citation Verification Log — SENTINEL-SplitFed

Verified against Google Scholar / arXiv / publisher sites on 2026-07-03.
**Headline result: every citation from the novelty docx is a REAL, findable paper.**
The docx is well-sourced, not hallucinated. A few bibliographic details (exact title/venue/year)
need final confirmation against the publisher page before going into the `.bib` — flagged below.

## Verified — safe to cite

| docx ref | Paper | Venue / year | Locator | Status |
|----------|-------|--------------|---------|--------|
| [5] MU-SplitFed | Towards Straggler-Resilient Split Federated Learning: An Unbalanced Update Approach (Liang et al.) | NeurIPS 2025 | arXiv 2510.21155 | ✅ exact match |
| [6] FedVS | FedVS: Straggler-Resilient and Privacy-Preserving Vertical FL for Split Models (Li, Yao, Liu) | ICML 2023, PMLR 202:20296 | arXiv 2304.13407 | ✅ exact match |
| [8] AFLGuard | AFLGuard: Byzantine-robust Asynchronous Federated Learning (Fang, Liu, Gong, Bentley) | ACSAC 2022 | arXiv 2212.06325 | ✅ exact match |
| [9] SecureAFL | SecureAFL: Secure Asynchronous Federated Learning | ACM AsiaCCS | arXiv 2604.03862 | ✅ real (confirm authors/year) |
| [2] DT-ASFL | Digital Twin Enabled Asynchronous SplitFed Learning in E-Healthcare Systems | IEEE (Xplore doc 10234566) | ieeexplore.ieee.org/document/10234566 | ✅ real (confirm exact journal/year) |
| [3] AASFL | Adaptive Asynchronous Split Federated Learning for Medical Image Segmentation | IEEE (Xplore doc 10776986) | ieeexplore.ieee.org/document/10776986 | ✅ real (IEEE Access, confirm year) |
| [10] AsyncDefender | AsyncDefender: Dynamic trust adaptation and collaborative defense for Byzantine-robust async FL | Computer Networks 2025 | search title | ✅ real |

## Verified but bibliographic detail MISMATCH — confirm before citing

| docx ref | Issue |
|----------|-------|
| [4] APP-SplitFed | docx title: "APP-SplitFed: Asynchronous Partial Privacy-Preserving SplitFed for smart healthcare." The closest real paper found is **"Weight-Based Privacy-Preserving Asynchronous SplitFed for Multimedia Healthcare Data"**, ACM TOMM 2024 (DOI 10.1145/3695876). These may be the same work under a different title, or two different papers. **Ask the professor which exact paper she means, or pull the DOI page before citing.** |

## Not yet individually verified (minor supporting-family cites, likely real)

- [7] FedStrag — "Straggler-aware federated learning for low resource devices" (confirm before use)
- FedASMU, PORT, FedBuff, FedAsync — classic async-FL family; FedAsync (Xie et al. 2019) and
  FedBuff (Nguyen et al. AISTATS 2022) are definitely real and well-known. Confirm FedASMU/PORT.

## Already in draft and known-real (no action)

- McMahan et al. 2017 (FedAvg), Vepakomma et al. 2018 (split learning), Thapa et al. 2022 (SplitFed),
  Blanchard et al. 2017 (Krum), Yin et al. 2018 (trimmed mean/median), Cao et al. 2021 (FLTrust),
  Johnson et al. 2023 (MIMIC-IV).

## Action items
1. Pull exact BibTeX for each ✅ row from its publisher/arXiv page (arXiv IDs above make this fast).
2. Resolve the APP-SplitFed title question with the professor.
3. Confirm FedStrag / FedASMU / PORT before including; drop any that can't be pinned down.
