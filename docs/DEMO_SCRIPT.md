# NeuroBridge: demo script (about 6 minutes)

Audience: an Owkin researcher or recruiter. Setup: run `streamlit run app/streamlit_app.py` with data cached. Open the Overview page. Optionally keep a finished run open in a second tab.

---

**0:00 · The problem (Overview page)**
> "Over the last ten years, Michelle Monje's and Frank Winkler's labs showed that neurons form synapses onto glioma cells and drive tumour growth. That's mechanistic work in mice. The question a pharma team asks next is translational. Is this biology visible in patients' molecular data? Does it define subgroups, does it track outcome, and does it replicate? NeuroBridge is an agentic platform that answers questions like that on public data, and is honest about what the evidence supports."

Point at the central question, the data-availability numbers, and "all public data, no synthetic patients".

**0:45 · Launch (click "Run the demo investigation")**
> "I'm asking: is neuronal activity-related signalling associated with GBM molecular subtypes and survival?"

While it runs (~1 min), narrate the live stage panel:
- **Hypothesis Agent.** "It decomposes the question into four hypotheses with explicit nulls. It also adds two workflows I didn't ask for: a composition control and single-cell localisation. A bulk neuronal signal is uninterpretable without them."
- **Literature Agent.** "Citations come from Europe PMC live, by PMID. The model never types a reference."
- **Data Agent.** "TCGA for discovery, two CGGA cohorts for validation, Darmanis single-cell. Each file is hashed into a manifest."
- **Bioinformatics, Statistics and Validation agents.** "Every line here is a real tool call with typed inputs, retries and a logged result."

**2:00 · Conclusions tab**
Walk the graded claims in order:
1. **S1 (exploratory).** "In TCGA the score is highest in Proneural and lowest in Mesenchymal; Hedges g is about −0.9."
2. **S2 (confirmatory).** "It replicates in both independent CGGA cohorts. CGGA has no subtype labels, so the Validation Agent trained a subtype classifier on TCGA only, with the candidate genes removed from its features so it can't manufacture the result. Cross-validated balanced accuracy is 0.82."
3. **C1 and S3 (the twist).** "But the score correlates 0.7 to 0.8 with mature-neuron markers. Residualise on them and the subtype difference no longer replicates. So the platform tells me this signal is largely normal brain in the sample, not proven tumour biology."
4. **O2.** "Survival: not replicated. TCGA is null, one CGGA cohort is significant in the *opposite* direction, and the reviewer flags that as heterogeneity, not support."
5. **SP1–SP3 (spatial).** "On one Visium section, presynaptic VGLUT1 sits with the neurons, but the receptor genes are spread through tumour tissue regardless of neuronal content. GAP43 and NTRK2 rise near neuron-rich tissue even after adjusting for each spot's own neuron content. VGLUT1 to glutamate receptors and BDNF to TrkB co-localise beyond what shared tissue domains explain. It's one patient, so this is a lead, and the platform labels it that way."
6. **SC1.** "In single cells, these genes are mostly in OPCs, neurons and astrocytes. GRIA1 is the one enriched in neoplastic cells, and VGLUT1 is absent from tumour cells, as you'd expect for a presynaptic marker."

> "The headline is a well-supported negative. Bulk transcript signatures are the wrong readout for neuron–glioma coupling. That is exactly the kind of decision-relevant answer you want before spending on a biomarker programme."

**3:30 · Transparency (Agents and Execution-log tabs)**
Open one agent: show its inputs, actions, outputs and limitations. Open one tool call: show the JSON with attempts and duration. Point at the evidence ceiling in the Plan tab: "statistical association, never mechanism".

**4:15 · Validation Lab**
- **Survival models.** Switch to "single gene → BDNF". Add or remove covariates. Show the Schoenfeld PH table. "Assumptions are tested, not assumed."
- **Cross-cohort validation.** Flip the residualisation toggle and watch the verdict change live.
- **Reproducibility.** Seed, package versions, and SHA-256 for every input file.

**5:00 · Biological Explorer (optional)**
Show the Spatial tab: domain map on the H&E image, then GRIA2 vs SNAP25 maps side by side. Then the single-cell UMAP coloured by GRIA2. Say the caveat out loud: co-localisation is not interaction.

**5:30 · Evidence Report**
Scroll the 13 sections, then download the PDF.
> "Every sentence is tagged as association, mechanistic-from-literature, plausibility or insufficient, and links to an analysis id or a PMID."

**5:50 · Architecture page and close**
> "No agent framework: a transparent orchestrator, 17 typed tools, and a registry that exports an MCP manifest, so these skills could plug into a K-Pro-style agent tomorrow. The LLM is optional and confined to language tasks. It never produces a number. Next steps would be deconvolution against single-cell references and a multi-patient spatial cohort, which is where I'd take this question."

Optional extra (with Claude enabled): ask "Is NLGN3 or BDNF expression associated with overall survival in IDH-wildtype glioblastoma after adjusting for age and MGMT status?" The planner restricts to IDH-wildtype and tests each gene alone and across cohorts. Claude's review narrative then critiques the conclusions using only the computed claims.

---

### Likely questions

- **"Why not let the LLM run the analysis?"** Reproducibility and auditability. The LLM plans and critiques; deterministic, tested code computes. Each step is replayable from the run folder.
- **"How do you avoid leakage?"**
  - Discovery and validation are separate institutions, and patient overlap is checked by a test.
  - The classifier sees only TCGA, and candidate genes are excluded from its features.
  - Scores are standardised within cohort.
  - Samples are never pooled.
- **"What if a dataset is down?"** The executor retries. The Data Agent records the gap and its consequence, and the reviewer lists it. Tests break a loader on purpose to check this.
- **"Statistical choices?"**
  - Effect sizes with CIs everywhere, and BH correction within families.
  - Per-SD hazard ratios with fixed reference levels.
  - Treatment variables are not adjusted for, because they are post-diagnosis.
  - Events-per-variable and PH assumptions are checked.
