# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:28:43 2025

@author: User
"""

import os
import numpy as np
import pandas as pd
import mne
import traceback

# === Load subject-session pairs ===
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\EEG DATA\EEG Logbook.xlsx"
df = pd.read_excel(file_path, header=2)

# Rename for easier access
df = df.rename(columns={
    'Subject id': 'SUB',
    'Opties: L2/R1/R2 (L2 en R1 zijn voor EEG hetzelfde, het hangt er vanaf of de pp ook in de longitudinale studie participeert)': 'SESSION'
})

subject_session_array = df[df['SUB'] != 'SUB'][['SUB', 'SESSION']].dropna().values

# === Define channel list and result structure ===
frontocentral_chs = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz',
                     'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FT10',
                     'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'Cz']

columns = ['SUB', 'SESSION']
for ch in frontocentral_chs:
    columns.append(f'ITPC_085_{ch}')
    columns.append(f'ITPC_256_{ch}')
    columns.append(f'ITPC_085_SURR_{ch}')
    columns.append(f'ITPC_256_SURR_{ch}')

result_df = pd.DataFrame(np.nan, index=range(subject_session_array.shape[0]), columns=columns)

# === Loop over participants ===
for i, (subject, session) in enumerate(subject_session_array):
    SUBnr = str(subject).strip()
    SESnr = str(session).strip().upper()
    result_df.loc[i, 'SUB'] = SUBnr
    result_df.loc[i, 'SESSION'] = SESnr

    print(f"\n📌 Subject: {SUBnr}, Session: '{SESnr}'")

    file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_preproc_eeg.fif"
    events_file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_events.npy"
    working_dir = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\EEG DATA\Processed\temp"
    fpath = os.path.join(working_dir, file)
    events_path = os.path.join(working_dir, events_file)

    if not os.path.exists(fpath) or not os.path.exists(events_path):
        print(f"❌ File not found: {file} or {events_file}")
        continue

    try:
        raw = mne.io.read_raw_fif(fpath, preload=True)
        events = np.load(events_path)
        sfreq = raw.info['sfreq']

        epochs = mne.Epochs(raw, events, event_id={'word': 1}, tmin=0, tmax=1.17,
                            baseline=None, preload=True, reject_by_annotation=True)
        cleaned_data = epochs.get_data()
        continuous_data = cleaned_data.transpose(1, 0, 2).reshape(epochs.info['nchan'], -1)
        cleaned_raw = mne.io.RawArray(continuous_data, epochs.info)

        starts = events[events[:, 2] == 12][:, 0]
        stops = events[events[:, 2] == 10][:, 0]
        if stops[0] < starts[0]:
            stops = stops[1:]

        block_epochs = []
        for j in range(len(starts)):
            selected = np.where((epochs.events[:, 0] >= starts[j]) & (epochs.events[:, 0] <= stops[j]))[0]
            block_epochs.append(epochs[selected])

        block1, block2, block3 = block_epochs[:3]

        def create_long_epochs(block, group_size=10):
            data = block.get_data()
            n_epochs, n_channels, n_times = data.shape
            n_full_groups = n_epochs // group_size
            trimmed_data = data[:n_full_groups * group_size]
            reshaped = trimmed_data.reshape(n_full_groups, group_size, n_channels, n_times)
            long_data = reshaped.transpose(0, 2, 1, 3).reshape(n_full_groups, n_channels, group_size * n_times)
            return mne.EpochsArray(long_data, block.info.copy())

        grouped_epochs = mne.concatenate_epochs([
            create_long_epochs(block1),
            create_long_epochs(block2),
            create_long_epochs(block3)
        ])

        # Surrogate data
        epoch_duration_sec = 11.7
        epoch_duration_samples = int(epoch_duration_sec * sfreq)
        raw_data = cleaned_raw.get_data()
        rng = np.random.default_rng(seed=42)
        start_samples = rng.integers(0, raw_data.shape[1] - epoch_duration_samples, size=len(grouped_epochs))
        surrogate_data = np.stack([raw_data[:, s:s + epoch_duration_samples] for s in start_samples])
        surrogate_epochs = mne.EpochsArray(surrogate_data, raw.info)

        # Pick frontocentral channels
        available_chs = [ch for ch in frontocentral_chs if ch in grouped_epochs.ch_names]
        grouped_epochs.pick_channels(available_chs)
        surrogate_epochs.pick_channels(available_chs)

        # Method: FFT-ITPC on real data
        data = grouped_epochs.get_data()
        fft = np.fft.rfft(data, axis=2)
        phases = np.angle(fft)
        itc = np.abs(np.mean(np.exp(1j * phases), axis=0))
        freqs = np.fft.rfftfreq(data.shape[2], d=1/sfreq)
        idx_085 = np.argmin(np.abs(freqs - 1/(0.39*3)))
        idx_256 = np.argmin(np.abs(freqs - 1/0.39))

        for ch_idx, ch in enumerate(available_chs):
            result_df.loc[i, f'ITPC_085_{ch}'] = itc[ch_idx, idx_085]
            result_df.loc[i, f'ITPC_256_{ch}'] = itc[ch_idx, idx_256]

        # FFT-ITPC on surrogate data
        data_surr = surrogate_epochs.get_data()
        fft_surr = np.fft.rfft(data_surr, axis=2)
        phases_surr = np.angle(fft_surr)
        itc_surr = np.abs(np.mean(np.exp(1j * phases_surr), axis=0))

        for ch_idx, ch in enumerate(available_chs):
            result_df.loc[i, f'ITPC_085_SURR_{ch}'] = itc_surr[ch_idx, idx_085]
            result_df.loc[i, f'ITPC_256_SURR_{ch}'] = itc_surr[ch_idx, idx_256]

        print(f"✅ Done: {SUBnr}-{SESnr}")

    except Exception as e:
        print(f"❌ Error processing {SUBnr}-{SESnr}: {e}")
        traceback.print_exc()
        continue

# Save to Excel
output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\ITPCPerChannel.xlsx"
result_df.to_excel(output_path, index=False)
print(f"✅ Saved ITPC values per channel to: {output_path}")
clean_df = result_df.dropna(subset=result_df.columns.difference(['SUB', 'SESSION']), how='all')



import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_test
from mne.channels import find_ch_adjacency

# === STEP 1: Load ITPC per-channel data ===
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\ITPCPerChannel.xlsx"  # <--- UPDATE this path
df = pd.read_excel(file_path)

# === STEP 2: Extract relevant column names ===
electrodes = [col.replace('ITPC_085_', '') for col in df.columns if col.startswith('ITPC_085_') and not col.startswith('ITPC_085_SURR_')]
real_columns = [f"ITPC_085_{ch}" for ch in electrodes]
surr_columns = [f"ITPC_085_SURR_{ch}" for ch in electrodes]

# Remove rows with NaNs (optional but recommended)
df_clean = df.dropna(subset=real_columns + surr_columns)

# Extract data matrices
X_real = df_clean[real_columns].to_numpy()
X_surr = df_clean[surr_columns].to_numpy()

print(f"✅ Loaded data with shape: {X_real.shape}")

# === STEP 3: Load an EEG file to build adjacency ===
example_fif = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\EEG DATA\Processed\temp\TULIP_SUB7_L2_EEG_preproc_eeg.fif"
raw = mne.io.read_raw_fif(example_fif, preload=False)
raw.pick_channels(electrodes)  # Match ordering to your data columns
adjacency, ch_names_used = find_ch_adjacency(raw.info, ch_type='eeg')
print(f"✅ Adjacency shape: {adjacency.shape}")

# === STEP 4: Run the cluster-based permutation test ===
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    [X_real, X_surr],
    n_permutations=1000,
    tail=1,  # Test if real > surrogate
    adjacency=adjacency,
    out_type='mask',
    seed=42,
    n_jobs=1
)

# === STEP 5: Print significant results ===
alpha = 0.05
print("\n=== Significant Clusters (p < 0.05) ===")
any_sig = False
for i, p_val in enumerate(cluster_p_values):
    if p_val < alpha:
        cluster_indices = np.where(clusters[i])[0]
        sig_channels = [electrodes[idx] for idx in cluster_indices]
        print(f"→ Cluster {i+1}: p = {p_val:.4f}, channels: {sig_channels}")
        any_sig = True

if not any_sig:
    print("No significant clusters found.")



import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords

# === Step 1: Compute mean difference ===
mean_diff = X_real.mean(axis=0) - X_surr.mean(axis=0)

# === Step 2: Load 2D positions ===
raw = mne.io.read_raw_fif(example_fif, preload=False)
raw.pick_channels(electrodes)
pos_2d = _find_topomap_coords(raw.info, picks='eeg')

# === Step 3: Get cluster mask ===
sig_cluster_idx = np.where(cluster_p_values < 0.05)[0][0]
cluster_mask = clusters[sig_cluster_idx]  # shape: (n_channels,)

# === Step 4: Plot base topomap ===
fig, ax = plt.subplots(figsize=(6, 6))
im, _ = mne.viz.plot_topomap(
    mean_diff,
    pos=pos_2d,
    axes=ax,
    cmap='Reds',
    contours=0,
    vlim=(np.min(mean_diff), np.max(mean_diff)),
    sensors=False,  # We'll overlay sensors ourselves
    show=False
)

# === Step 5: Overlay markers ===
# Plot all channels as small gray markers
ax.scatter(
    pos_2d[:, 0], pos_2d[:, 1],
    color='lightgray', s=30, zorder=2, edgecolor='k'
)

# Plot significant channels as red markers on top
sig_pos = pos_2d[cluster_mask]
ax.scatter(
    sig_pos[:, 0], sig_pos[:, 1],
    color='crimson', edgecolor='black', s=60, zorder=3
)

# === Finalize ===
ax.set_title('ITPC: Real – Surrogate at 0.85 Hz\n', fontsize=12)
plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, label='ΔITPC')
plt.tight_layout()
plt.show()

















import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_test
from mne.channels import find_ch_adjacency

# === STEP 1: Load ITPC per-channel data ===
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\ITPCPerChannel.xlsx"  # <--- UPDATE this path
df = pd.read_excel(file_path)

# === STEP 2: Extract relevant column names ===
electrodes = [col.replace('ITPC_256_', '') for col in df.columns if col.startswith('ITPC_256_') and not col.startswith('ITPC_256_SURR_')]
real_columns = [f"ITPC_256_{ch}" for ch in electrodes]
surr_columns = [f"ITPC_256_SURR_{ch}" for ch in electrodes]

# Remove rows with NaNs (optional but recommended)
df_clean = df.dropna(subset=real_columns + surr_columns)

# Extract data matrices
X_real = df_clean[real_columns].to_numpy()
X_surr = df_clean[surr_columns].to_numpy()

print(f"✅ Loaded data with shape: {X_real.shape}")

# === STEP 3: Load an EEG file to build adjacency ===
example_fif = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\EEG DATA\Processed\temp\TULIP_SUB7_L2_EEG_preproc_eeg.fif"
raw = mne.io.read_raw_fif(example_fif, preload=False)
raw.pick_channels(electrodes)  # Match ordering to your data columns
adjacency, ch_names_used = find_ch_adjacency(raw.info, ch_type='eeg')
print(f"✅ Adjacency shape: {adjacency.shape}")

# === STEP 4: Run the cluster-based permutation test ===
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    [X_real, X_surr],
    n_permutations=1000,
    tail=1,  # Test if real > surrogate
    adjacency=adjacency,
    out_type='mask',
    seed=42,
    n_jobs=1
)

# === STEP 5: Print significant results ===
alpha = 0.05
print("\n=== Significant Clusters (p < 0.05) ===")
any_sig = False
for i, p_val in enumerate(cluster_p_values):
    if p_val < alpha:
        cluster_indices = np.where(clusters[i])[0]
        sig_channels = [electrodes[idx] for idx in cluster_indices]
        print(f"→ Cluster {i+1}: p = {p_val:.4f}, channels: {sig_channels}")
        any_sig = True

if not any_sig:
    print("No significant clusters found.")






import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords

# === Step 1: Compute mean difference ===
mean_diff = X_real.mean(axis=0) - X_surr.mean(axis=0)

# === Step 2: Load 2D positions ===
raw = mne.io.read_raw_fif(example_fif, preload=False)
raw.pick_channels(electrodes)
pos_2d = _find_topomap_coords(raw.info, picks='eeg')

# === Step 3: Get cluster mask ===
sig_cluster_idx = np.where(cluster_p_values < 0.05)[0][0]
cluster_mask = clusters[sig_cluster_idx]  # shape: (n_channels,)

# === Step 4: Plot base topomap ===
fig, ax = plt.subplots(figsize=(6, 6))
im, _ = mne.viz.plot_topomap(
    mean_diff,
    pos=pos_2d,
    axes=ax,
    cmap='Reds',
    contours=0,
    vlim=(np.min(mean_diff), np.max(mean_diff)),
    sensors=False,  # We'll overlay sensors ourselves
    show=False
)

# === Step 5: Overlay markers ===
# Plot all channels as small gray markers
ax.scatter(
    pos_2d[:, 0], pos_2d[:, 1],
    color='lightgray', s=30, zorder=2, edgecolor='k'
)

# Plot significant channels as red markers on top
sig_pos = pos_2d[cluster_mask]
ax.scatter(
    sig_pos[:, 0], sig_pos[:, 1],
    color='crimson', edgecolor='black', s=60, zorder=3
)

# === Finalize ===
ax.set_title('ITPC 2.56 Hz: Real - Surrogate\n', fontsize=12)
plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, label='ΔITPC')
plt.tight_layout()
plt.show()






















