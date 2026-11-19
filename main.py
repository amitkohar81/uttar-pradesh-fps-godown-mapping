import os
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# PIPELINE CONFIGURATION & INPUT ASSIGNMENTS (SETTLEMENT-SIZE HIERARCHY)
# ==============================================================================
RAW_BOUNDARY_FILE = "final_recovered_matched_up.geojson"
GODOWNS_CSV = "Godowns_Deduplicated_LocationDetails.csv"
FPS_CSV = "FPS_Deduplicated_LocationDetails.csv"
BENEFICIARIES_CSV = "Beneficiaries_LocationDetails.csv"

# Institutional Population-Group Thresholds (Total Urban Town Beneficiaries)
# 100,000 = Official Census of India Class-I City & RBI Urban vs. Semi-Urban Cutoff
POP_THRESHOLDS = [50000, 75000, 100000, 150000, 200000]
BASELINE_THRESHOLD = 100000

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print("======================================================================")
print(" STARTING SETTLEMENT-SIZE TRANSPORT PIPELINE & LATEX GENERATOR")
print("======================================================================\n")

# Load Raw Spatial Inputs Once
raw_gdf = gpd.read_file(RAW_BOUNDARY_FILE)
boundary_crs = raw_gdf.crs
raw_gdf['Area_Type'] = raw_gdf['Area_Type'].replace({'Semi-Urban': 'Urban'})


def assign_boundaries_robust(points_df, lat_col, long_col, boundaries_clean):
    """Point-in-polygon mapping with a 5km nearest-neighbor fallback."""
    df = points_df.copy()
    df['temp_point_id'] = range(len(df))

    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[long_col], df[lat_col]),
        crs="EPSG:4326"
    )

    tagged = gpd.sjoin(points_gdf, boundaries_clean, how="left", predicate="within")
    tagged = tagged.drop_duplicates(subset=['temp_point_id'], keep='first')
    tagged = tagged.drop(columns=['index_right'], errors='ignore')

    unmatched_mask = tagged['Town_Block_Name'].isna()
    if unmatched_mask.sum() > 0:
        unmatched_gdf = points_gdf[points_gdf['temp_point_id'].isin(tagged.loc[unmatched_mask, 'temp_point_id'])]
        unmatched_proj = unmatched_gdf.to_crs(epsg=32644)
        boundaries_proj = boundaries_clean.to_crs(epsg=32644)

        nearest_matches = gpd.sjoin_nearest(
            unmatched_proj,
            boundaries_proj,
            how="left",
            max_distance=5000,
            distance_col="dist_to_boundary_m"
        )
        nearest_matches = nearest_matches.drop_duplicates(subset=['temp_point_id'], keep='first')

        cols_to_fill = [c for c in ['District', 'Area_Type', 'Town_Block_Name'] if c in nearest_matches.columns]
        for col in cols_to_fill:
            mapping_dict = dict(zip(nearest_matches['temp_point_id'], nearest_matches[col]))
            tagged.loc[unmatched_mask, col] = tagged.loc[unmatched_mask, 'temp_point_id'].map(mapping_dict)

    return tagged.drop(columns=['geometry', 'temp_point_id', 'index_right'], errors='ignore')


def get_distance_bucket(dist_km):
    """Categorize supply distance into 4 operational haul tiers."""
    if pd.isna(dist_km):
        return "Unknown"
    elif dist_km < 5.0:
        return "Within 5 km"
    elif 5.0 <= dist_km < 15.0:
        return "5 km - 15 km"
    elif 15.0 <= dist_km <= 30.0:
        return "15 km - 30 km"
    else:
        return "Beyond 30 km"


# ==============================================================================
# STAGE 1: INVARIANT SPATIAL TAGGING, B_i ALLOCATION & GODOWN DISTANCES
# ==============================================================================
print("Stage 1: Assigning spatial boundaries, B_i allocations, and nearest-godown distances...")

base_boundaries_4326 = raw_gdf.to_crs(epsg=4326)[['District', 'Area_Type', 'Town_Block_Name', 'geometry']].copy()
base_boundaries_4326['District_Clean'] = base_boundaries_4326['District'].astype(str).str.strip().str.upper()
base_boundaries_4326['Town_Block_Name_Clean'] = base_boundaries_4326['Town_Block_Name'].astype(str).str.strip().str.upper()

raw_godowns = pd.read_csv(GODOWNS_CSV)
raw_fps = pd.read_csv(FPS_CSV)

tagged_godowns = assign_boundaries_robust(raw_godowns, "From_Lat", "From_Long", base_boundaries_4326)
tagged_fps_base = assign_boundaries_robust(raw_fps, "To_Lat", "To_Long", base_boundaries_4326)

fps_base_df = tagged_fps_base.dropna(
    subset=['District', 'Area_Type', 'Town_Block_Name', 'To_Lat', 'To_Long']
).copy()
fps_base_df = fps_base_df.rename(columns={'Area_Type': 'Base_Area_Type'})
godowns_df = tagged_godowns.dropna(subset=['District', 'From_Lat', 'From_Long']).copy()
ben_df = pd.read_csv(BENEFICIARIES_CSV)

for df_tmp, sec_col in [(ben_df, 'Area_Type'), (fps_base_df, 'Base_Area_Type')]:
    df_tmp['Area_Type_Clean'] = df_tmp[sec_col].replace({'Semi-Urban': 'Urban'}).astype(str).str.strip().str.upper()
    df_tmp['Town_Block_Name_Clean'] = df_tmp['Town_Block_Name'].astype(str).str.strip().str.upper()
    df_tmp['District_Clean'] = df_tmp['District'].astype(str).str.strip().str.upper()

# Sector-Isolated Beneficiary Disaggregation (100% invariant across all thresholds)
ben_sector_summary = ben_df.groupby(['District_Clean', 'Town_Block_Name_Clean', 'Area_Type_Clean'])[
    'Total_Beneficiary'].sum().reset_index()
town_sector_fps = fps_base_df.groupby(['District_Clean', 'Town_Block_Name_Clean', 'Area_Type_Clean']).size().reset_index(
    name='Town_Sector_FPS_Count')

ben_merged = pd.merge(ben_sector_summary, town_sector_fps,
                      on=['District_Clean', 'Town_Block_Name_Clean', 'Area_Type_Clean'], how='left')
matched_sector = ben_merged[ben_merged['Town_Sector_FPS_Count'] > 0].copy()
orphan_sector = ben_merged[
    ben_merged['Town_Sector_FPS_Count'].isna() | (ben_merged['Town_Sector_FPS_Count'] == 0)].copy()

matched_sector['Beneficiary_Per_FPS'] = matched_sector['Total_Beneficiary'] / matched_sector['Town_Sector_FPS_Count']

fps_base_df = pd.merge(
    fps_base_df,
    matched_sector[['District_Clean', 'Town_Block_Name_Clean', 'Area_Type_Clean', 'Beneficiary_Per_FPS']],
    on=['District_Clean', 'Town_Block_Name_Clean', 'Area_Type_Clean'],
    how='left'
)
fps_base_df['Beneficiary_Per_FPS'] = fps_base_df['Beneficiary_Per_FPS'].fillna(0.0)

if len(orphan_sector) > 0:
    orphan_dist_summary = orphan_sector.groupby(['District_Clean', 'Area_Type_Clean'])[
        'Total_Beneficiary'].sum().reset_index(name='Orphan_Beneficiaries')
    district_sector_fps = fps_base_df.groupby(['District_Clean', 'Area_Type_Clean']).size().reset_index(
        name='District_Sector_FPS_Count')

    orphan_dist_merged = pd.merge(orphan_dist_summary, district_sector_fps,
                                  on=['District_Clean', 'Area_Type_Clean'], how='left')
    orphan_dist_merged['Add_Ben_Per_FPS'] = (
        orphan_dist_merged['Orphan_Beneficiaries'] / orphan_dist_merged['District_Sector_FPS_Count']
    ).fillna(0.0)

    fps_base_df = pd.merge(
        fps_base_df,
        orphan_dist_merged[['District_Clean', 'Area_Type_Clean', 'Add_Ben_Per_FPS']],
        on=['District_Clean', 'Area_Type_Clean'],
        how='left'
    )
    fps_base_df['Beneficiary_Per_FPS'] += fps_base_df['Add_Ben_Per_FPS'].fillna(0.0)
    fps_base_df = fps_base_df.drop(columns=['Add_Ben_Per_FPS'])

# Compute Aggregate Urban Settlement Size (Total Beneficiaries per Statutory Town)
urban_town_pop = (
    fps_base_df[fps_base_df['Base_Area_Type'] == 'Urban']
    .groupby(['District_Clean', 'Town_Block_Name_Clean'])['Beneficiary_Per_FPS']
    .sum()
    .reset_index(name='Town_Urban_Beneficiaries')
)
fps_base_df = pd.merge(
    fps_base_df, urban_town_pop, on=['District_Clean', 'Town_Block_Name_Clean'], how='left'
)
fps_base_df['Town_Urban_Beneficiaries'] = fps_base_df['Town_Urban_Beneficiaries'].fillna(0.0)

# Statewide Closest Godown Distances via cKDTree (EPSG:32644)
fps_base_gdf = gpd.GeoDataFrame(
    fps_base_df, geometry=gpd.points_from_xy(fps_base_df['To_Long'], fps_base_df['To_Lat']), crs="EPSG:4326"
).to_crs(epsg=32644)
godowns_gdf = gpd.GeoDataFrame(
    godowns_df, geometry=gpd.points_from_xy(godowns_df['From_Long'], godowns_df['From_Lat']), crs="EPSG:4326"
).to_crs(epsg=32644)

tree_state = cKDTree(np.column_stack([godowns_gdf.geometry.x, godowns_gdf.geometry.y]))
min_dists_state, state_indices = tree_state.query(np.column_stack([fps_base_gdf.geometry.x, fps_base_gdf.geometry.y]), k=1)

fps_base_gdf['Statewide_Distance_KM'] = (min_dists_state / 1000.0).round(2)
fps_base_gdf['Statewide_Closest_Godown_Code'] = godowns_gdf.iloc[state_indices]['From_ID'].values
fps_base_gdf['Statewide_Closest_Godown_District'] = godowns_gdf.iloc[state_indices]['District'].values

# Distance per 1,000 Beneficiaries: (d_i / B_i) * 1000
fps_base_gdf['Dist_Per_1k_Ben_KM'] = np.where(
    fps_base_gdf['Beneficiary_Per_FPS'] > 0,
    (fps_base_gdf['Statewide_Distance_KM'] / fps_base_gdf['Beneficiary_Per_FPS']) * 1000.0,
    np.nan
)

plot_tracking_data = []
baseline_buckets_df = None
baseline_map_gdf = None

# ==============================================================================
# STAGE 2: INSTITUTIONAL SETTLEMENT-SIZE SENSITIVITY LOOP (50k TO 200k)
# ==============================================================================
for pop_cutoff in POP_THRESHOLDS:
    cutoff_k_label = f"{int(pop_cutoff // 1000)}k"
    print(f"Processing Institutional Town Cutoff >= {pop_cutoff:,} Beneficiaries ({cutoff_k_label})...")

    df_run = fps_base_gdf.copy()
    is_rural = df_run['Base_Area_Type'] == 'Rural'
    is_class1_urban = (~is_rural) & (df_run['Town_Urban_Beneficiaries'] >= pop_cutoff)

    df_run['Area_Type'] = np.where(
        is_rural, 'Rural',
        np.where(is_class1_urban, 'Urban', 'Semi-Urban')
    )
    df_run['Zone_Subtype'] = np.where(
        is_rural, 'Rural Sector',
        np.where(is_class1_urban, f'Class-I Urban (>= {cutoff_k_label})', f'Semi-Urban Town (< {cutoff_k_label})')
    )

    if pop_cutoff == BASELINE_THRESHOLD:
        detailed_export = pd.DataFrame({
            'FPS_ID': df_run['To_ID'],
            'District': df_run['District'],
            'Town_Block_Name': df_run['Town_Block_Name'],
            'Area_Type': df_run['Area_Type'],
            'Zone_Subtype': df_run['Zone_Subtype'],
            'FPS_Latitude': df_run['To_Lat'],
            'FPS_Longitude': df_run['To_Long'],
            'Derived_Beneficiaries': df_run['Beneficiary_Per_FPS'].round(2),
            'Statewide_Closest_Godown_Code': df_run['Statewide_Closest_Godown_Code'],
            'Statewide_Closest_Godown_District': df_run['Statewide_Closest_Godown_District'],
            'Statewide_Distance_KM': df_run['Statewide_Distance_KM'],
            'Dist_Per_1k_Ben_KM': df_run['Dist_Per_1k_Ben_KM'].round(2)
        })
        detailed_export.to_csv("FPS_Detailed_Location_and_Godown_Mapping_ClassI_100k.csv", index=False)

        # Build baseline map GeoDataFrame
        map_poly = base_boundaries_4326.merge(
            urban_town_pop, on=['District_Clean', 'Town_Block_Name_Clean'], how='left'
        )
        map_poly['Town_Urban_Beneficiaries'] = map_poly['Town_Urban_Beneficiaries'].fillna(0.0)
        map_poly['Plot_Sector'] = np.where(
            map_poly['Area_Type'] == 'Rural', 'Rural',
            np.where(map_poly['Town_Urban_Beneficiaries'] >= pop_cutoff, 'Urban', 'Semi-Urban')
        )
        baseline_map_gdf = map_poly.copy()

    segments = ['Urban', 'Semi-Urban', 'Rural']
    tii_rows = []

    urban_sub = df_run[df_run['Area_Type'] == 'Urban']
    urban_ben = urban_sub['Beneficiary_Per_FPS'].sum()
    urban_pkm = (urban_sub['Beneficiary_Per_FPS'] * urban_sub['Statewide_Distance_KM']).sum()
    urban_tii_base = (urban_pkm / urban_ben) if urban_ben > 0 else 1.0

    for seg in segments:
        seg_sub = df_run[df_run['Area_Type'] == seg]
        n_fps = len(seg_sub)
        tot_ben = seg_sub['Beneficiary_Per_FPS'].sum()

        # 1. Mean Drop Distance (km)
        mean_drop_dist = seg_sub['Statewide_Distance_KM'].mean()

        # 2. Distance per 1,000 Beneficiaries (km / 1k ben)
        mean_dist_per_1k = seg_sub['Dist_Per_1k_Ben_KM'].mean()

        # 3. Tonnage-Weighted Haul Distance (TII, km) & Multiplier
        tot_pkm = (seg_sub['Beneficiary_Per_FPS'] * seg_sub['Statewide_Distance_KM']).sum()
        tii_km = (tot_pkm / tot_ben) if tot_ben > 0 else np.nan
        tii_mult = (tii_km / urban_tii_base) if urban_tii_base > 0 else np.nan

        tii_rows.append({
            'Cutoff_Label': f"$\\ge$ {cutoff_k_label}",
            'Cutoff_Clean': f">={cutoff_k_label}",
            'Area_Type': seg,
            'Total_FPS_Count': n_fps,
            'Total_Beneficiaries': int(round(tot_ben)),
            'Mean_Drop_Dist_KM': round(mean_drop_dist, 2),
            'Dist_Per_1k_Ben_KM': round(mean_dist_per_1k, 2),
            'Tonnage_Weighted_Haul_KM': round(tii_km, 2),
            'Haul_Multiplier': round(tii_mult, 2)
        })

    plot_tracking_data.append(pd.DataFrame(tii_rows))

    # Compute Operational Haul-Radius Distribution at 100k Census Class-I Baseline
    if pop_cutoff == BASELINE_THRESHOLD:
        df_run['Distance_Bucket'] = df_run['Statewide_Distance_KM'].apply(get_distance_bucket)
        valid_fps_df = df_run[df_run['Distance_Bucket'] != "Unknown"].copy()
        bucket_order = ["Within 5 km", "5 km - 15 km", "15 km - 30 km", "Beyond 30 km"]
        bucket_rows = []

        totals_by_type = {
            a: {
                'fps': len(valid_fps_df[valid_fps_df['Area_Type'] == a]),
                'ben': valid_fps_df[valid_fps_df['Area_Type'] == a]['Beneficiary_Per_FPS'].sum()
            }
            for a in segments
        }

        for bucket in bucket_order:
            row = {'Distance_Bucket': bucket}
            for area_type in segments:
                seg_df = valid_fps_df[valid_fps_df['Area_Type'] == area_type]
                b_fps = seg_df[seg_df['Distance_Bucket'] == bucket]
                b_count = len(b_fps)
                b_ben = b_fps['Beneficiary_Per_FPS'].sum()

                row[f"{area_type}_FPS_Count"] = b_count
                row[f"{area_type}_FPS_%"] = round((b_count / totals_by_type[area_type]['fps'] * 100.0), 2) if totals_by_type[area_type]['fps'] > 0 else 0.0
                row[f"{area_type}_Beneficiaries_%"] = round((b_ben / totals_by_type[area_type]['ben'] * 100.0), 2) if totals_by_type[area_type]['ben'] > 0 else 0.0

            bucket_rows.append(row)

        baseline_buckets_df = pd.DataFrame(bucket_rows)
        baseline_buckets_df.to_csv("Service_Radius_Coverage_Buckets_ClassI_100k.csv", index=False)

df_master_tracking = pd.concat(plot_tracking_data, ignore_index=True)
df_master_tracking.to_csv("Statewide_Settlement_Sensitivity_50k_to_200k.csv", index=False)

# ==============================================================================
# STAGE 3: AUTOMATED NATIVE LATEX TABLE GENERATOR
# ==============================================================================
print("\nGenerating native LaTeX tables: table1_sensitivity.tex & table2_buckets.tex...")

tex_t1_path = "table1_sensitivity.tex"
with open(tex_t1_path, "w") as f:
    f.write(r"\begin{table}[htbp]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Statewide Transporter Haul-Distance Gradient Across Institutional Settlement Tiers ($\tau = 50\text{k}$ to $200\text{k}$; Census Class-I Benchmark $= 100\text{k}$).}" + "\n")
    f.write(r"\label{tab:transporter_sensitivity}" + "\n")
    f.write(r"\small" + "\n")
    f.write(r"\resizebox{\linewidth}{!}{%" + "\n")
    f.write(r"\begin{tabular}{c l r r c c c c}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(
        r"\textbf{City Cutoff ($\tau$)} & \textbf{Settlement Tier} & \textbf{FPS Stops} & \textbf{Beneficiaries} & "
        r"\textbf{Mean Drop Dist ($\bar{D}$)} & \textbf{Dist / 1k Ben ($\bar{D}_{\text{1k}}$)} & "
        r"\textbf{Tonnage Haul ($\text{TII}$)} & \textbf{Haul Multiplier} \\" + "\n"
    )
    f.write(r"\midrule" + "\n")

    cutoffs_unique = df_master_tracking['Cutoff_Label'].unique()
    for idx, c in enumerate(cutoffs_unique):
        sub = df_master_tracking[df_master_tracking['Cutoff_Label'] == c]
        f.write(r"\multirow{3}{*}{\textbf{" + c + r"}}" + "\n")

        for _, r in sub.iterrows():
            sec = r['Area_Type']
            fps = f"{int(r['Total_FPS_Count']):,}"
            ben = f"{int(r['Total_Beneficiaries']):,}"
            d_mean = f"{r['Mean_Drop_Dist_KM']:.2f} km"
            d_1k = f"{r['Dist_Per_1k_Ben_KM']:.2f} km"
            tii_d = f"{r['Tonnage_Weighted_Haul_KM']:.2f} km"
            tii_m = f"{r['Haul_Multiplier']:.2f} $\\times$"

            f.write(
                f"& {sec:<10} & {fps:>6} & {ben:>12} & {d_mean:>8} & {d_1k:>8} & {tii_d:>8} & {tii_m:>8} \\\\\n"
            )

        if idx < len(cutoffs_unique) - 1:
            f.write(r"\midrule" + "\n")

    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}%" + "\n")
    f.write(r"}" + "\n")
    f.write(r"\end{table}" + "\n")

if baseline_buckets_df is not None:
    tex_t2_path = "table2_buckets.tex"
    with open(tex_t2_path, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Operational Haul-Radius Distribution Across Settlement Tiers at Census Class-I Benchmark ($\tau = 100\text{k}$, Normalized to 100.00\%).}" + "\n")
        f.write(r"\label{tab:coverage_buckets}" + "\n")
        f.write(r"\small" + "\n")
        f.write(r"\resizebox{\linewidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{l c r r c r r c r r}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(
            r"\multirow{2}{*}{\textbf{Haul Distance Tier}} & \multicolumn{3}{c}{\textbf{Urban (Class-I Cities)}} & \multicolumn{3}{c}{\textbf{Semi-Urban (Towns)}} & \multicolumn{3}{c}{\textbf{Rural Sector}} \\" + "\n"
        )
        f.write(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}" + "\n")
        f.write(
            r"& \textbf{FPS Stops} & \textbf{FPS \%} & \textbf{Load \%} & \textbf{FPS Stops} & \textbf{FPS \%} & \textbf{Load \%} & \textbf{FPS Stops} & \textbf{FPS \%} & \textbf{Load \%} \\" + "\n"
        )
        f.write(r"\midrule" + "\n")

        for _, r in baseline_buckets_df.iterrows():
            tier = r['Distance_Bucket']
            u_fps, u_fps_p, u_ben_p = f"{int(r['Urban_FPS_Count']):,}", rf"{r['Urban_FPS_%']:.2f}\%", rf"{r['Urban_Beneficiaries_%']:.2f}\%"
            su_fps, su_fps_p, su_ben_p = f"{int(r['Semi-Urban_FPS_Count']):,}", rf"{r['Semi-Urban_FPS_%']:.2f}\%", rf"{r['Semi-Urban_Beneficiaries_%']:.2f}\%"
            r_fps, r_fps_p, r_ben_p = f"{int(r['Rural_FPS_Count']):,}", rf"{r['Rural_FPS_%']:.2f}\%", rf"{r['Rural_Beneficiaries_%']:.2f}\%"

            f.write(
                f"\\textbf{{{tier}}} & {u_fps:>5} & {u_fps_p:>7} & {u_ben_p:>7} & {su_fps:>5} & {su_fps_p:>7} & {su_ben_p:>7} & {r_fps:>6} & {r_fps_p:>7} & {r_ben_p:>7} \\\\\n"
            )

        f.write(r"\midrule" + "\n")
        tot_u_fps = f"{int(baseline_buckets_df['Urban_FPS_Count'].sum()):,}"
        tot_su_fps = f"{int(baseline_buckets_df['Semi-Urban_FPS_Count'].sum()):,}"
        tot_r_fps = f"{int(baseline_buckets_df['Rural_FPS_Count'].sum()):,}"

        f.write(
            f"\\textbf{{Total Mapped}} & \\textbf{{{tot_u_fps}}} & \\textbf{{100.00\\%}} & \\textbf{{100.00\\%}} & \\textbf{{{tot_su_fps}}} & \\textbf{{100.00\\%}} & \\textbf{{100.00\\%}} & \\textbf{{{tot_r_fps}}} & \\textbf{{100.00\\%}} & \\textbf{{100.00\\%}} \\\\\n"
        )
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")

# ==============================================================================
# STAGE 4: PUBLICATION-GRADE VISUALIZATIONS (FIGURES 1 TO 4)
# ==============================================================================
print("Rendering high-resolution experimental figures (1 to 4)...")

# FIGURE 1: LINE PLOT (Mean Drop Distance vs Distance per 1k Beneficiaries)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
cutoffs_clean = ['>=50k', '>=75k', '>=100k', '>=150k', '>=200k']
colors = {'Urban': '#1f4e78', 'Semi-Urban': '#ff7f0e', 'Rural': '#2b5c8f'}
markers = {'Urban': '^', 'Semi-Urban': 's', 'Rural': 'o'}

for seg in ['Urban', 'Semi-Urban', 'Rural']:
    sub = df_master_tracking[df_master_tracking['Area_Type'] == seg]
    ax1.plot(cutoffs_clean, sub['Mean_Drop_Dist_KM'], marker=markers[seg], color=colors[seg], linewidth=2.5, label=seg)
    ax2.plot(cutoffs_clean, sub['Dist_Per_1k_Ben_KM'], marker=markers[seg], color=colors[seg], linewidth=2.5, label=seg)

ax1.set_title("A: Mean Delivery Distance per Stop ($\\bar{D}$)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Class-I City Population Threshold ($\\tau$)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Mean Distance to Godown (km)", fontsize=10, fontweight='bold')
ax1.legend(frameon=True, loc='right')

ax2.set_title("B: Load-Normalized Distance per 1,000 Beneficiaries ($\\bar{D}_{\\text{1k}}$)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Class-I City Population Threshold ($\\tau$)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Distance per 1,000 Beneficiaries (km)", fontsize=10, fontweight='bold')
ax2.legend(frameon=True, loc='right')

plt.suptitle("Statewide Haul-Distance Gradient Across Institutional Settlement Tiers", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("Figure1_Distance_Variation_By_Area_Type.png", dpi=300)
plt.close()

# FIGURE 2: CONSOLIDATED BARS (50k vs 100k Class-I Benchmark)
fig, ax = plt.subplots(figsize=(9, 5))
df_sub = df_master_tracking[df_master_tracking['Cutoff_Clean'].isin(['>=50k', '>=100k'])]
x = np.arange(3)
width = 0.35

vals_50 = [df_sub[(df_sub['Cutoff_Clean'] == '>=50k') & (df_sub['Area_Type'] == s)]['Mean_Drop_Dist_KM'].values[0] for s in ['Urban', 'Semi-Urban', 'Rural']]
vals_100 = [df_sub[(df_sub['Cutoff_Clean'] == '>=100k') & (df_sub['Area_Type'] == s)]['Mean_Drop_Dist_KM'].values[0] for s in ['Urban', 'Semi-Urban', 'Rural']]

rects1 = ax.bar(x - width / 2, vals_50, width, label='>=50k City Threshold', color='#1f4e78')
rects2 = ax.bar(x + width / 2, vals_100, width, label='>=100k Census Class-I Benchmark', color='#ff7f0e')
ax.set_title("Monotonic Haul-Distance Escalation Across Settlement Tiers", fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Urban (Class-I Cities)', 'Semi-Urban (Towns)', 'Rural Sector'], fontsize=11, fontweight='bold')
ax.set_ylabel("Mean Delivery Distance per Stop (km)", fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='upper left')

for rect in rects1 + rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f} km', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("Figure2_Consolidated_Experiment_Summary.png", dpi=300)
plt.close()

# FIGURE 3: BUCKET COVERAGE DISTRIBUTION (AT 100k CENSUS CLASS-I BENCHMARK)
fig, ax = plt.subplots(figsize=(9, 5))
tiers = ["Within 5 km", "5 km - 15 km", "15 km - 30 km", "Beyond 30 km"]
urban_pcts = [baseline_buckets_df.loc[baseline_buckets_df['Distance_Bucket'] == t, 'Urban_Beneficiaries_%'].values[0] for t in tiers]
semi_pcts = [baseline_buckets_df.loc[baseline_buckets_df['Distance_Bucket'] == t, 'Semi-Urban_Beneficiaries_%'].values[0] for t in tiers]
rural_pcts = [baseline_buckets_df.loc[baseline_buckets_df['Distance_Bucket'] == t, 'Rural_Beneficiaries_%'].values[0] for t in tiers]

x = np.arange(len(tiers))
width = 0.25
rects1 = ax.bar(x - width, urban_pcts, width, label='Urban (Class-I Cities)', color='#1f4e78')
rects2 = ax.bar(x, semi_pcts, width, label='Semi-Urban (Towns)', color='#ff7f0e')
rects3 = ax.bar(x + width, rural_pcts, width, label='Rural Sector', color='#4a777a')

ax.set_title("Share of Grain Load by Haul-Distance Tier (Census Class-I Benchmark)", fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tiers, fontsize=10, fontweight='bold')
ax.set_ylabel("Share of Sector Load (%)", fontsize=10, fontweight='bold')
ax.legend(frameon=True, loc='upper right')

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig("Figure3_Coverage_Radius_Distribution.png", dpi=300)
plt.close()

# FIGURE 4: STATEWIDE ZONAL MAP (WHITE RURAL CANVAS AT 100k BENCHMARK)
fig, ax = plt.subplots(figsize=(11, 8))
color_map = {'Rural': '#ffffff', 'Semi-Urban': '#ff7f0e', 'Urban': '#1f4e78'}

for area_type in ['Rural', 'Semi-Urban', 'Urban']:
    sub_gdf = baseline_map_gdf[baseline_map_gdf['Plot_Sector'] == area_type]
    sub_gdf.plot(
        ax=ax,
        color=color_map[area_type],
        edgecolor='#cccccc' if area_type == 'Rural' else '#555555',
        linewidth=0.2 if area_type == 'Rural' else 0.4
    )

ax.set_title("Statewide Spatial Distribution of Institutional Settlement Tiers in Uttar Pradesh\n(Census Class-I City Benchmark >= 100,000 Beneficiaries)",
             fontsize=12, fontweight='bold', pad=14)
ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight='bold')
ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.4)

legend_elements = [
    mpatches.Patch(facecolor='#ffffff', edgecolor='#cccccc', label='Rural Sector'),
    mpatches.Patch(facecolor='#ff7f0e', edgecolor='#555555', label='Semi-Urban Statutory Towns (< 100k)'),
    mpatches.Patch(facecolor='#1f4e78', edgecolor='#555555', label='Urban Class-I Cities (>= 100k)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9.5)

plt.tight_layout()
plt.savefig("Figure4_Statewide_Zonal_Boundaries.png", dpi=300)
plt.close()

print("\n======================================================================")
print(" PIPELINE COMPLETE: M&SOM TABLES & FIGURES SUCCESSFULLY GENERATED")
print("======================================================================\n")
