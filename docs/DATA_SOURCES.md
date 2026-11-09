# Data sources

The numbers below were checked against the sources on 2026-11-09 and are recomputed from the files at load time. Every download is recorded in `data/cache/manifest.json` with its URL, SHA-256, size and retrieval time.

## Ingested

### TCGA-GBM RNA-seq: discovery
- **Files.** UCSC Xena TCGA hub, `TCGA.GBM.sampleMap/HiSeqV2` (log2(RSEM+1), 20,530 genes × 172 samples) and `TCGA.GBM.sampleMap/GBM_clinicalMatrix`.
  - `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.GBM.sampleMap%2FHiSeqV2.gz`
  - `https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.GBM.sampleMap%2FGBM_clinicalMatrix`
- **Samples.** 154 primary tumours, 13 recurrent and 5 normal.
- **Survival.** Overall survival is `days_to_death` for deceased patients, otherwise `days_to_last_followup`, divided by 30.4375.
- **Subtype labels.** `GeneExp_Subtype` gives the Verhaak 2010 labels: Mesenchymal 55, Classical 42, Proneural 39, Neural 28.
- **IDH proxy.** G-CIMP status (`G_CIMP_STATUS`) is used, because IDH status is not in this matrix.
- **Publication.** Brennan et al. 2013, *Cell*, PMID 24120142. Subtypes from Verhaak et al. 2010, PMID 20129251.

### CGGA mRNAseq_325 and mRNAseq_693: validation
- **Source.** Chinese Glioma Genome Atlas, release 2020-05-06 (`http://www.cgga.org.cn/download.jsp`). Expression and clinical zips are downloaded directly.
- **Expression.** RSEM gene estimates; NeuroBridge applies log2(x+1).
- **Clinical file quirks.** The files use CR line endings and contain stray whitespace (for example `"Female "`). Both are normalised.
- **GBM selection.** Histology GBM, rGBM or sGBM:
  - CGGA-325: 139 samples, of which 85 are primary GBM.
  - CGGA-693: 249 samples, of which 140 are primary GBM.
- **Variables.** OS (days), censor, age, sex, IDH, 1p/19q, MGMT, radiotherapy, temozolomide. No Verhaak labels are distributed; NeuroBridge projects them, see the methodology.
- **Independence.** No patient overlap with TCGA or between batches. This is checked in tests.
- **Publication.** Zhao et al. 2021, *Genomics Proteomics Bioinformatics*, PMID 33662628.

### GEO GSE84465: single-cell reference
- **Files.** `GSE84465_GBM_All_data.csv.gz` (HTSeq counts, 23,460 genes × 3,589 cells) and the series matrix (author cell types, patient, tumour vs periphery).
- **Cells.** Immune 1,847; Neoplastic 1,091; OPC 406; Astrocyte 88; Oligodendrocyte 85; Vascular 51; Neuron 21. The GEO label "Astocyte" is corrected to "Astrocyte".
- **Publication.** Darmanis et al. 2017, *Cell Reports*, PMID 29091775.

### 10x Visium Human Glioblastoma: spatial reference
- **Files.** 10x Genomics dataset "Human Glioblastoma: Whole Transcriptome Analysis" (Space Ranger 1.2.0, licence CC BY 4.0), about 30 MB:
  - `https://cf.10xgenomics.com/samples/spatial-exp/1.2.0/Parent_Visium_Human_Glioblastoma/Parent_Visium_Human_Glioblastoma_filtered_feature_bc_matrix.h5`
  - `..._spatial.tar.gz` (H&E images, spot positions, scale factors), extracted with tarfile's `data` filter
  - `..._metrics_summary.csv`
- **Content.** 3,468 tissue spots × 36,601 genes from one section of one patient. No clinical annotation; raw reads withheld by 10x for donor anonymity.
- **Caveat.** A single case: results are descriptive.

### TCGA-GBM U133A array: platform check only
- **File.** Xena `TCGA.GBM.sampleMap/HT_HG-U133A`, 12,042 genes × 539 samples.
- **Caveat.** Same patients as TCGA RNA-seq, so it is never used as validation.

### Gene sets
- **Libraries.** Enrichr `MSigDB_Hallmark_2020` (50 sets), `KEGG_2021_Human` (320), `GO_Biological_Process_2023` (5,406) and `Reactome_2022` (1,816). Retrieved via GSEApy and cached as JSON with a timestamp.

### Literature
- **Europe PMC REST API**, MEDLINE records only. The 16 curated key references are stored as PMIDs only; all metadata is fetched live.

## Listed, not ingested

| Resource | Why not ingested |
|---|---|
| cBioPortal (gbm_tcga_pub2013 and others) | Queried live for listings and targeted gene/clinical lookups; the API is not designed for full matrices, and Xena provides the same TCGA data |
| CELLxGENE glioma collections | Listed live with cell counts and file sizes; h5ad files are 1.2–7.3 GB |
| GSE131928 (Neftel 2019) | ~1 GB RAW tar of per-sample matrices; cell-state scores must be reconstructed |
| GSE16011 (Gravendeel 2009) | Survival is not deposited in GEO |
| GBM spatial cohort (Ravi 2022, Visium, multiple patients) | Multi-GB archive; the next step to test the spatial observations across patients |

## Demonstration data

The application uses no demonstration data. Synthetic cohorts are generated only inside `tests/` (`make_synthetic_cohort`) to check that estimators recover known effects. They are flagged `is_demo=True` and never displayed.
