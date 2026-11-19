# Uttar Pradesh PDS Settlement-Size & Haul-Distance Pipeline (`up-pds-spatial-analysis`)

This repository contains the Python pipeline (`main.py`) and geospatial datasets used to classify institutional settlement tiers (Urban, Semi-Urban, Rural) and compute Public Distribution System (PDS) transport haul-distance metrics across Uttar Pradesh, India. 

Using administrative boundaries, Fair Price Shop (FPS) coordinates, Food Corporation of India (FCI) godown locations, and beneficiary counts, the script performs sensitivity analyses across population thresholds (τ = 50k to 200k, benchmarked at the Census of India Class-I threshold of 100k) and automatically outputs publication-ready LaTeX tables and high-resolution figures.

## Repository Structure

All input files reside in the root directory so the pipeline runs out of the box:

```text
up-pds-spatial-analysis/
├── README.md                                       # Project documentation
├── CITATION.cff                                    # GitHub citation metadata
├── LICENSE                                         # Dual MIT (code) & CC-BY 4.0 (data) license
├── requirements.txt                                # Python package dependencies
├── main.py                                         # Main analysis script
├── final_recovered_matched_up.geojson              # Urban & rural administrative polygons
├── Godowns_Deduplicated_LocationDetails.csv        # FCI storage godown coordinates
├── FPS_Deduplicated_LocationDetails.csv            # Active Fair Price Shop (FPS) coordinates
└── Beneficiaries_LocationDetails.csv               # District/town/sector beneficiary totals
```

## Methodology Overview

1. **Spatial Boundary Assignment (`EPSG:32644`):**
   * Geographic boundaries, active FPS outlets, and FCI godowns are projected onto `EPSG:32644` (WGS 84 / UTM zone 44N) for planar distance calculations.
   * Each active FPS outlet $i$ and godown is assigned to its parent administrative polygon via point-in-polygon intersection (`within`), supplemented by a **5 km nearest-polygon fallback** (`sjoin_nearest`) for boundary-edge coordinates.
2. **Sector-Isolated Beneficiary Disaggregation:**
   * Census beneficiary totals are disaggregated proportionally across active FPS outlets strictly within matching district, block/town, and binary sector units ($B_i$). Orphaned sector beneficiaries are redistributed across active FPS outlets within the matching district and sector.
3. **Nearest-Godown Distance Metrics:**
   * Statewide closest FCI godown Euclidean distances ($d_i$) are computed using `scipy.spatial.cKDTree`.
   * Outputs include Mean Drop Distance ($\bar{D}$), Load-Normalized Distance per 1,000 Beneficiaries ($\bar{D}_{\text{1k}}$), and Tonnage-Weighted Haul Distance ($\text{TII}$).
4. **Institutional Settlement Hierarchy:**
   * Statutory urban towns are partitioned into **Class-I Urban** ($\ge \tau$) and **Semi-Urban** ($< \tau$) tiers across thresholds $\tau \in \{50\text{k}, 75\text{k}, 100\text{k}, 150\text{k}, 200\text{k}\}$.

## Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd up-pds-spatial-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Generated Outputs

Running `main.py` generates the following datasets, tables, and figures in the root directory:
* **CSV Datasets:** `FPS_Detailed_Location_and_Godown_Mapping_ClassI_100k.csv`, `Service_Radius_Coverage_Buckets_ClassI_100k.csv`, and `Statewide_Settlement_Sensitivity_50k_to_200k.csv`.
* **LaTeX Tables:** `table1_sensitivity.tex` and `table2_buckets.tex`.
* **Figures (300 DPI PNG):** `Figure1_Distance_Variation_By_Area_Type.png` through `Figure4_Statewide_Zonal_Boundaries.png`.

## License

* **Code:** Licensed under the [MIT License](LICENSE).
* **Datasets:** Licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org).

## Citation

If you use this code or dataset in your research, please cite our working paper and repository:

```bibtex
@misc{VikasEtAl2026_UP_PDS,
  author       = {Vikas, Kumar and Raz, G. and Kohar, Amit and Jakhar, S. K.},
  title        = {[Fleet Electrification in Public Food Distribution: Carbon Credit Incentives vs. Transition Mandates]},
  year         = {2026},
  howpublished = {Working Paper},
  url          = {https://github.com}
}
```
