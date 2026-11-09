# Scientific methodology

## Framing

The question has three parts: subgroups, neuron–tumour interactions and outcomes. Each maps to a testable claim with an explicit null:

| ID | Hypothesis | Null | Data | Type |
|---|---|---|---|---|
| H1 | The candidate score differs between transcriptional subtypes | identical distributions | TCGA (discovery), CGGA ×2 (validation) | exploratory in TCGA, confirmatory in CGGA |
| H2 | The candidate score is associated with overall survival after adjustment | adjusted HR per SD = 1 | TCGA, CGGA ×2 | confirmatory |
| H3 | Candidate genes are expressed by neoplastic cells | not higher in neoplastic than in other cells | GSE84465 | exploratory |
| H5 | In tissue, candidate genes extend beyond neuron-rich regions and neuron–glioma pairs co-localise | expression tracks neuronal content only; co-localisation no higher than shared domains predict | one Visium section | exploratory, descriptive |
| H4 | Bulk associations are not explained by normal-brain content | effects vanish after neuron-marker adjustment | all bulk cohorts | sensitivity |

**Evidence ceiling.** These are observational transcriptomic data without perturbation or longitudinal sampling. The highest attainable level is *statistical association*. Mechanistic statements come only from the cited experimental literature and are tagged as literature claims.

## Gene panels

- **Candidate panel (pre-specified by the brief).** NLGN3, SLC17A7, GRIA1, GRIA2, GRIN2B, BDNF. Each has a stated rationale and citation. They are investigation targets, not biomarkers.
- **Extended panel (assembled by NeuroBridge and labelled as such).** Adds NTRK2, GRIA3/4, GRIN1, THBS1, GAP43 and GJA1.
- **Negative-control panel.** Mature-neuron markers SNAP25, SYT1, RBFOX3, GABRA1, NEFL, STMN2, SYN1 and GAD1. In bulk tissue, a "neuronal activity" signature may simply measure how much normal cortex was resected with the tumour. This panel is what lets the platform test that alternative.
- **Pathway sets.** KEGG Glutamatergic synapse; GO Chemical synaptic transmission; GO Regulation of neuronal synaptic plasticity.

## Scores

- **Score.** For each cohort separately, z-score each gene across primary tumours, then average over the panel's measured genes.
- **Scale.** Scores are on an SD scale within a cohort and are never compared in absolute terms across cohorts. Cohorts are never pooled at the sample level, which avoids batch artefacts between platforms.
- **Robustness.** A rank-based score (mean within-sample percentile) is used in sensitivity analyses.

## Group comparisons

- **Tests.** Kruskal-Wallis with ε² as effect size. Pairwise Mann-Whitney with Benjamini–Hochberg correction across pairs. Hedges' g with a large-sample 95% CI, plus Cliff's δ.
- **Minimum group size.** Groups with n < 5 are excluded and reported.

## Subtype transfer to CGGA (leakage control)

CGGA distributes no Verhaak labels. To test H1 in independent data:

1. **Training.** Train only on TCGA primary tumours: Proneural, Classical and Mesenchymal. Neural is excluded because it is attributed to normal-tissue contamination (Wang et al. 2017) and would be circular with neuronal signal.
2. **Features.** Use the 1,500 most variable genes shared by all cohorts, after **removing every candidate, extended-panel, pathway and neuron-marker gene**. The classifier therefore cannot create an association with the score.
3. **Scaling.** z-score within each cohort, which removes platform location and scale.
4. **Model.** L2 multinomial logistic regression (C = 0.05, balanced classes). 5-fold stratified CV in TCGA gives a balanced accuracy of 0.82.
5. **Application.** Apply the model unchanged to CGGA. Validation cohorts contribute nothing to training or feature selection.
6. **Restriction.** Test only IDH-wildtype CGGA tumours, and non-G-CIMP TCGA tumours, which is where the subtypes are defined.

**Pre-specification note.** The Mesenchymal-vs-Proneural contrast was chosen after looking at TCGA. That is why TCGA results are labelled exploratory, while CGGA provides the prospective test.

## Differential expression and enrichment

- **Filtering.** Keep genes with mean log2 expression ≥ 1 that are detected in ≥ 20% of samples. This keeps the multiple-testing burden realistic.
- **Test.** Welch t-test, high vs low score (median split; tertiles optional), with BH FDR. Hedges' g is reported for every gene.
- **Circularity.** Score-defined groups make the panel genes differential by construction. This is stated on the figure.
- **ORA.** Hypergeometric test against the **measured-gene background**, not the whole genome. A whole-genome background inflates enrichment for brain-restricted sets.
- **GSEA.** Pre-ranked GSEA (GSEApy) on Hedges' g, 1,000 permutations, set size 15–500, seeded.

## Survival

- **Outcome.** Overall survival in months, primary tumours only, complete-case.
- **Kaplan–Meier.** Median or tertile split, Greenwood CIs, log-rank test. Shown for orientation; dichotomisation loses power.
- **Cox PH (lifelines, Efron ties).**
  - Continuous score standardised, so HR is per SD.
  - Categorical covariates dummy-coded against fixed reference levels (Male, IDH-wildtype, unmethylated MGMT, non-G-CIMP, Mesenchymal), so HRs read the same way in every cohort.
  - Covariates with more than 50% missing are dropped and reported.
  - Columns that become constant after missing-data exclusion are dropped. This avoids singular fits, which happened in CGGA-693 before the fix.
- **Assumption checks.**
  - Scaled Schoenfeld residual test (rank time transform) for each covariate. Violations are flagged, and the HR is read as time-averaged.
  - Events-per-variable below 10 is flagged.
- **Confounders.** Age, sex, IDH (CGGA) or G-CIMP (TCGA), MGMT (CGGA), and subtype where available. Radiotherapy and temozolomide are deliberately *not* adjusted for. They follow diagnosis and depend on performance status, so adjusting for them risks conditioning on a mediator or collider.

## Validation rule

- **Replication.** A hypothesis is replicated in a cohort when the estimate has the same sign as discovery and p < 0.05 in that independent cohort.
- **Verdicts.** "Replicated in k/K", "partially replicated", "direction consistent but not significant", "not replicated", or "not validated (no independent cohort analysed)". The last verdict is forced whenever only one cohort is available; this is tested.
- **Pooling.** DerSimonian–Laird random-effects pooling with Cochran's Q and I² is reported as *descriptive*, because it includes the discovery cohort.
- **Opposite direction.** A significant effect in the opposite direction is reported as heterogeneity, never as support.

## Sensitivity analyses

These are the survival specifications:
- unadjusted;
- rank score;
- adjustment for the neuron-marker score;
- score residualised on neuron markers;
- IDH-wildtype only (CGGA) or non-G-CIMP only (TCGA);
- Neural subtype excluded;
- top vs bottom tertile log-rank.

They are robustness checks, not new hypotheses, and are not multiplicity-corrected. H1 is also re-tested on the residualised score.

## Single cell

- **QC.** Keep cells with at least 500 genes and less than 20% mitochondrial reads; keep genes detected in at least 3 cells.
- **Processing.** normalise_total (1e4), log1p, 2,000 HVGs (Seurat flavour, batch-aware by patient), 30 PCs, kNN with k = 15, UMAP and Leiden (resolution 0.5), all seeded.
- **Labels.** Cell types are the authors' annotations.
- **Tests.** Wilcoxon rank-sum (scanpy) with BH correction. Neoplastic is compared with all other cells, or with a chosen reference such as OPC, and periphery with core within neoplastic cells.
- **Pseudoreplication.** Cells are not independent; with four patients a pseudobulk test would have no power. The caveat is attached to every result.
- **Ligand–receptor.** The score is mean ligand in the sender × mean receptor in the receiver, tested against a cell-label permutation null (one-sided, BH). Pairs are curated from neuron–glioma biology. This is co-expression, not evidence of interaction.

## Spatial transcriptomics

- **Data.** One 10x Visium GBM section (one patient). Every spatial result is a case-level description.
- **Preprocessing.** Spots with at least 500 UMIs and 250 genes; normalise_total (1e4), log1p, 2,000 HVGs, 30 PCs, kNN, Leiden domains (resolution 0.6), all seeded.
- **Programmes.** scanpy `score_genes` for mature-neuron, tumour/reactive-glial, myeloid and vascular sets. Each domain is labelled by its most enriched programme; these are proxies, not cell-type calls.
- **Spatial structure.** Moran's I on the Visium hexagonal neighbour graph, row-standardised, with a two-sided permutation p (999 permutations).
- **Interface analysis.**
  - Neuron-rich spots are the top 20% of the neuron programme.
  - For all other spots, ring distance to the nearest neuron-rich spot is computed by breadth-first search on the hex grid.
  - Spearman correlation of expression with distance, also as a partial correlation controlling for the spot's own neuron score. Genes like GAP43 and NTRK2 are expressed by neurons too, so this guards against partial-volume effects.
  - Interface spots (1–2 rings) vs distal spots (5 or more rings) with Hedges' g.
- **Ligand–receptor co-localisation.**
  - Statistic: the mean over spots of ligand(spot) × mean receptor(six neighbours).
  - Two nulls. The global null permutes the ligand across all spots. The domain-stratified null permutes the ligand only within its spatial domain, so co-localisation must exceed what shared tissue composition produces.
  - Claims use the domain-stratified null with BH correction.
- **Limits.** Spots are not independent, so spot-level p-values outside the permutation tests are optimistic. Transcript co-localisation is not interaction.

## Question-aware analysis

- **Named genes.** Genes named in the question (and present in the data) become a `user_genes` panel. When four or fewer are named, each is also tested alone, including cross-cohort survival validation.
- **Restrictions.** "IDH-wildtype" in the question restricts survival and validation analyses to IDH-wildtype tumours, using non-G-CIMP as the proxy in TCGA. IDH and G-CIMP are then removed from the adjustment set.
- **Verdicts.** If validation cohorts disagree in direction, the verdict is "not replicated (inconsistent directions across cohorts)", never "direction consistent".

## Claim grading and review

Claims are built from results by templates, so no free-text generation touches numbers. Each claim has the following fields:
- `level`: plausibility, association, mechanistic, causal or insufficient;
- `kind`: exploratory, confirmatory, sensitivity, literature or interpretation;
- `analyses`: analysis ids;
- `citations`: verified PMIDs;
- `caveats`;
- `status`.

The reviewer then applies these rules:
- **Causal verbs.** No causal verb is allowed in a claim derived from observational analysis.
- **Citations.** Literature claims need verified citations, otherwise they are removed.
- **Traceability.** Every analytic claim needs an analysis id.
- **Single cohort.** A single-cohort run downgrades confirmatory claims.
- **Composition.** A correlation above rho 0.5 with neuron markers raises a major issue.
- **Other checks.** PH violations, projected-label error, pseudoreplication, and failed tools or datasets, each with its consequence.

## Known limitations

See the README and the per-analysis `limitations` fields. The main ones are composition confounding in bulk data, projected subtypes, cohort-specific adjustment sets, a small single-cell reference, and the absence of spatial or perturbation data.
