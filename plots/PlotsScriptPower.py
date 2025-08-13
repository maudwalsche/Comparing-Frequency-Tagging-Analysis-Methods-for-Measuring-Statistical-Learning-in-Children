#The raw data cant be shared but an excel file will be available to do the plots

# === Load subject-session pairs ===
import os
import numpy as np
import pandas as pd
import mne
import traceback

logbook_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\EEG DATA\EEG Logbook.xlsx"
df = pd.read_excel(logbook_path, header=2)

# Clean column names
df = df.rename(columns={
    'Subject id': 'SUB',
    'Opties: L2/R1/R2 (L2 en R1 zijn voor EEG hetzelfde, het hangt er vanaf of de pp ook in de longitudinale studie participeert)': 'SESSION'
})

subject_session_array = df[df['SUB'] != 'SUB'][['SUB', 'SESSION']].dropna().values

# === Define channels and output structure ===
frontocentral_chs = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6','C3','Cz','C4']

# Frequencies: 0.25 to 5 Hz (step 0.25), plus 0.8547 Hz and 2.5641 Hz
target_freqs = np.round(np.arange(0.25, 5.25, 0.25), 4).tolist()
target_freqs += [0.8547, 2.5641]
target_freqs = sorted(set(np.round(target_freqs, 4)))

# Define output columns
columns = ['SUB', 'SESSION']
for ch in frontocentral_chs:
    for f in target_freqs:
        columns.append(f'POW_{f:.4f}_{ch}')
        columns.append(f'POW_{f:.4f}_SURR_{ch}')

# Initialize results
result_df = pd.DataFrame(np.nan, index=range(subject_session_array.shape[0]), columns=columns)

# === Loop over participants ===
for i, (subject, session) in enumerate(subject_session_array):
    SUBnr = str(subject).strip()
    SESnr = str(session).strip().upper()
    result_df.loc[i, 'SUB'] = SUBnr
    result_df.loc[i, 'SESSION'] = SESnr

    print(f"\n📌 Processing Subject: {SUBnr}, Session: {SESnr}")

    file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_preproc_eeg.fif"
    events_file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_events.npy"
    working_dir = r"C:\Users\User\OneDrive - UGent\TULIP - Towards Understanding Learning In Preterms - Data (No personal identifiers!)\EEG\Processed"
    fpath = os.path.join(working_dir, file)
    events_path = os.path.join(working_dir, events_file)

    if not os.path.exists(fpath) or not os.path.exists(events_path):
        print(f"❌ Missing files: {file} or {events_file}")
        continue

    try:
        raw = mne.io.read_raw_fif(fpath, preload=True)
        events = np.load(events_path)
        sfreq = raw.info['sfreq']

        # === Epoching ===
        epochs = mne.Epochs(raw, events, event_id={'word': 1}, tmin=0, tmax=1.17,
                            baseline=None, preload=True, reject_by_annotation=True)

        # === Reconstruct cleaned raw from epochs ===
        cleaned_data = epochs.get_data()
        continuous_data = cleaned_data.transpose(1, 0, 2).reshape(epochs.info['nchan'], -1)
        cleaned_raw = mne.io.RawArray(continuous_data, epochs.info)

        # === Identify blocks ===
        starts = events[events[:, 2] == 12][:, 0]
        stops = events[events[:, 2] == 10][:, 0]
        if stops[0] < starts[0]:
            stops = stops[1:]

        block_epochs = []
        for j in range(min(len(starts), len(stops))):
            selected = np.where((epochs.events[:, 0] >= starts[j]) & (epochs.events[:, 0] <= stops[j]))[0]
            block_epochs.append(epochs[selected])

        if len(block_epochs) < 3:
            print(f"⚠️ Not enough blocks for subject {SUBnr}")
            continue

        # === Group epochs into longer trials ===
        def create_long_epochs(block, group_size=10):
            data = block.get_data()
            n_epochs, n_channels, n_times = data.shape
            n_full_groups = n_epochs // group_size
            trimmed = data[:n_full_groups * group_size]
            reshaped = trimmed.reshape(n_full_groups, group_size, n_channels, n_times)
            long_data = reshaped.transpose(0, 2, 1, 3).reshape(n_full_groups, n_channels, group_size * n_times)
            return mne.EpochsArray(long_data, block.info.copy())

        grouped_epochs = mne.concatenate_epochs([
            create_long_epochs(block_epochs[0]),
            create_long_epochs(block_epochs[1]),
            create_long_epochs(block_epochs[2])
        ])

        # === Generate surrogate epochs ===
        epoch_duration_sec = 11.7
        epoch_samples = int(epoch_duration_sec * sfreq)
        raw_data = cleaned_raw.get_data()

        n_surrogates = 10
        surrogate_data_list = []
        for k in range(n_surrogates):
            rng = np.random.default_rng(seed=42 + k)
            start_samples = rng.integers(0, raw_data.shape[1] - epoch_samples, size=len(grouped_epochs))
            surrogate_data = np.stack([
                raw_data[:, s:s + epoch_samples] for s in start_samples
            ])
            surrogate_data_list.append(surrogate_data)

        surrogate_data_array = np.array(surrogate_data_list)
        mean_surrogate_data = np.mean(surrogate_data_array, axis=0)
        surrogate_epochs = mne.EpochsArray(mean_surrogate_data, raw.info)

        # === Pick frontocentral channels ===
        available_chs = [ch for ch in frontocentral_chs if ch in grouped_epochs.ch_names]
        grouped_epochs.pick_channels(available_chs)
        surrogate_epochs.pick_channels(available_chs)

        # === Compute POWER using evoked FFT method ===
        evoked = grouped_epochs.average()
        data = evoked.data
        fft_data = np.fft.rfft(data, axis=1)
        freqs = np.fft.rfftfreq(data.shape[1], d=1/sfreq)
        amplitude = np.abs(fft_data)

        # Surrogate power
        evoked_surr = surrogate_epochs.average()
        data_surr = evoked_surr.data
        fft_data_surr = np.fft.rfft(data_surr, axis=1)
        amplitude_surr = np.abs(fft_data_surr)

        for ch_idx, ch in enumerate(available_chs):
            for f in target_freqs:
                f_idx = np.argmin(np.abs(freqs - f))
                result_df.loc[i, f'POW_{f:.4f}_{ch}'] = amplitude[ch_idx, f_idx]
                result_df.loc[i, f'POW_{f:.4f}_SURR_{ch}'] = amplitude_surr[ch_idx, f_idx]

        print(f"✅ Done: {SUBnr}-{SESnr}")

    except Exception as e:
        print(f"❌ Error processing {SUBnr}-{SESnr}: {e}")
        traceback.print_exc()
        continue

# === Compute average POWER across all frontocentral channels ===
avg_df = result_df[['SUB', 'SESSION']].copy()
avg_df = avg_df[~((avg_df['SUB'] == '2701') & (avg_df['SESSION'] == 'R1'))]
for f in target_freqs:
    freq_str = f"{f:.4f}"
    real_cols = [f'POW_{freq_str}_{ch}' for ch in frontocentral_chs if f'POW_{freq_str}_{ch}' in result_df.columns]
    surr_cols = [f'POW_{freq_str}_SURR_{ch}' for ch in frontocentral_chs if f'POW_{freq_str}_SURR_{ch}' in result_df.columns]
    avg_df[f'POW_{freq_str}_MEAN'] = result_df[real_cols].mean(axis=1, skipna=True)
    avg_df[f'POW_{freq_str}_SURR_MEAN'] = result_df[surr_cols].mean(axis=1, skipna=True)

# === Save averaged results ===
avg_output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\POWforPlots_Averaged.xlsx"
avg_df.to_excel(avg_output_path, index=False)
print(f"✅ Averaged POWER values saved to: {avg_output_path}")




import pandas as pd
import matplotlib.pyplot as plt

# === Load your averaged power file ===
avg_path = r"POWforPlots_Averaged.xlsx"
avg_df = pd.read_excel(avg_path)

# === Collect real & surrogate columns ===
real_cols = [c for c in avg_df.columns if c.startswith("POW_") and c.endswith("_MEAN") and "_SURR" not in c]
surr_cols = [c for c in avg_df.columns if c.startswith("POW_") and c.endswith("_SURR_MEAN")]

if not real_cols:
    raise ValueError("No real power columns found (pattern 'POW_*_MEAN').")
if not surr_cols:
    raise ValueError("No surrogate power columns found (pattern 'POW_*_SURR_MEAN').")

# === Map frequency -> column name for real & surrogate ===
def parse_freq(col):
    # expects e.g., POW_0.85_MEAN or POW_0.85_SURR_MEAN
    return float(col.split('_')[1])

real_map = {parse_freq(c): c for c in real_cols}
surr_map = {parse_freq(c): c for c in surr_cols}

# Align on common frequencies (in case of any mismatch)
common_freqs = sorted(set(real_map.keys()) & set(surr_map.keys()))
if not common_freqs:
    raise ValueError("No overlapping frequencies between real and surrogate columns.")

real_cols_sorted = [real_map[f] for f in common_freqs]
surr_cols_sorted = [surr_map[f] for f in common_freqs]

# === Plot ===
plt.figure(figsize=(10, 6))

# Plot each participant's real power in light gray
for i in range(len(avg_df)):
    plt.plot(common_freqs, avg_df.loc[i, real_cols_sorted].values, color='lightgray', alpha=0.7)

# Plot group averages
mean_real = avg_df[real_cols_sorted].mean()
plt.plot(common_freqs, mean_real, color='black', linewidth=2, label='Mean Power (real)')

mean_surr = avg_df[surr_cols_sorted].mean()
plt.plot(common_freqs, mean_surr, color='red', linewidth=2, linestyle='--', label='Mean Power (surrogate)')

# === Labels and style ===
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (Amplitude)")
plt.title("Power Across Frequencies (Real + Surrogate Mean)")
plt.xlim(left=0.5)
plt.ylim(top=0.025)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





