# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 13:55:48 2025

@author: User
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne import EpochsArray
from mne.time_frequency import tfr_morlet
import traceback
import itertools
from scipy.stats import pearsonr
from mne import EpochsArray
import numpy as np
from mne import concatenate_epochs



# === Load subject-session pairs ===
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\EEG DATA\EEG Logbook.xlsx"
df = pd.read_excel(file_path, header=2)

# Rename for easier use later
df = df.rename(columns={
    'Subject id': 'SUB',
    'Opties: L2/R1/R2 (L2 en R1 zijn voor EEG hetzelfde, het hangt er vanaf of de pp ook in de longitudinale studie participeert)': 'SESSION'
})

subject_session_array = df[['SUB', 'SESSION']].dropna().values
subject_session_array = df[df['SUB'] != 'SUB'][['SUB', 'SESSION']].dropna().values

# === Initialize results dataframe ===
result_df = pd.DataFrame(
    np.nan,
    index=range(subject_session_array.shape[0]),
    columns=[
        'SUB',
        'SESSION',
        'WLI_ITPC',
        'WLI_POWER',
        'WLI_ITPC_WAVELET',
        'ITPC_085',
        'ITPC_170',
        'ITPC_256',
        'ITPC_085_SURR',
        'ITPC_170_SURR',
        'ITPC_256_SURR',
        'POWER_085',
        'POWER_170',
        'POWER_256',
        'POWER_085_SURR',
        'POWER_170_SURR',
        'POWER_256_SURR',
        'WAVE_085',
        'WAVE_170',
        'WAVE_256',
        'WAVE_085_SURR',
        'WAVE_170_SURR',
        'WAVE_256_SURR'
    ]
)



# === Define expected event IDs ===
event_id = {
    'word': 1 }
missing_data = []

# === Loop over subject-session pairs ===
for i, (subject, session) in enumerate(subject_session_array):
    SUBnr = str(subject).strip()
    SESnr = str(session).strip().upper()
    result_df.loc[i, 'SUB'] = SUBnr
    result_df.loc[i, 'SESSION'] = SESnr

    print(f"\n📌 Subject: {SUBnr}, Session: '{SESnr}'")

    file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_preproc_eeg.fif"
    events_file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_events.npy"

    working_dir = r"C:\Users\User\OneDrive - UGent\TULIP - Towards Understanding Learning In Preterms - Data (No personal identifiers!)\EEG\Processed\0_used_for_manipulation_check"
    fpath = os.path.join(working_dir, file)
    events_path = os.path.join(working_dir, events_file)

    if not os.path.exists(fpath) or not os.path.exists(events_path):
        print(f"❌ File not found: {file} or {events_file}")
        result_df.loc[i] = result_df.loc[i] = [SUBnr, SESnr, 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND']
        missing_data.append((SUBnr, SESnr))
        continue

    try:
        print("✅ Loading EEG and events...")
        # Load data (already done above)
        raw = mne.io.read_raw_fif(fpath, preload=True)
        events = np.load(events_path)
        sfreq = raw.info['sfreq']

        # Create word-level epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id={'word': 1},
            tmin=0,
            tmax=1.17,
            baseline=None,
            preload=True,
            reject_by_annotation=True
        )
        cleaned_data = epochs.get_data()
        continuous_data = cleaned_data.transpose(1, 0, 2).reshape(epochs.info['nchan'], -1)
        cleaned_raw = mne.io.RawArray(continuous_data, epochs.info)



        
        # Get sample positions of structured block starts (12) and stops (10)
        starts = events[events[:, 2] == 12][:, 0]
        stops = events[events[:, 2] == 10][:, 0]

        # Ensure we pair only valid start-stop segments (skip first stop if it's before first start)
        if stops[0] < starts[0]:
            stops = stops[1:]

        # Loop through and slice epochs into blocks
        block_epochs = []

        for j in range(len(starts)):
            start_sample = starts[j]
            stop_sample = stops[j]

            # Select epochs whose event (sample) is within this range
            selected = np.where((epochs.events[:, 0] >= start_sample) &
                                (epochs.events[:, 0] <= stop_sample))[0]

            block = epochs[selected]
            block_epochs.append(block)

            print(f"Block {i+1}: {len(block)} epochs")

        # Now you can access each block as:
        block1 = block_epochs[0]
        block2 = block_epochs[1]
        block3 = block_epochs[2]
        def create_long_epochs(block, group_size=10):
            data = block.get_data()          # shape: (n_epochs, n_channels, n_times)
            n_epochs, n_channels, n_times = data.shape

            # Calculate how many full groups of 10 we can make
            n_full_groups = n_epochs // group_size
            n_used = n_full_groups * group_size

            # Trim to usable data only
            trimmed_data = data[:n_used]

            # Reshape to combine every 10 epochs
            reshaped = trimmed_data.reshape(n_full_groups, group_size, n_channels, n_times)

            # Concatenate time dimension within each group
            long_data = reshaped.transpose(0, 2, 1, 3).reshape(n_full_groups, n_channels, group_size * n_times)

            # Use metadata from original block
            info = block.info.copy()

            # Create new EpochsArray
            long_epochs = mne.EpochsArray(long_data, info)
            return long_epochs
        grouped_block1 = create_long_epochs(block1)
        grouped_block2 = create_long_epochs(block2)
        grouped_block3 = create_long_epochs(block3)



        # Combine all grouped blocks into a single Epochs object
        grouped_epochs = mne.concatenate_epochs([grouped_block1, grouped_block2, grouped_block3])

        # Optional: check result
        print(grouped_epochs)
        print(f"Total long epochs: {len(grouped_epochs)}")
        print(f"Duration of one long epoch: {grouped_epochs.times[-1] - grouped_epochs.times[0]:.2f} s")

        
        
        # --- Surrogate data
        epoch_duration_sec = 11.7
        sfreq = raw.info['sfreq']
        epoch_duration_samples = int(epoch_duration_sec * sfreq)
        
        # --- Generate surrogate epochs ---
        n_grouped_epochs = len(grouped_epochs)
        raw_data = cleaned_raw.get_data()
        n_channels, n_total_samples = raw_data.shape
        
        # Choose random start sample indices
        max_start_sample = n_total_samples - epoch_duration_samples
        rng = np.random.default_rng(seed=42)  # optional: seed for reproducibility
        start_samples = rng.integers(0, max_start_sample, size=n_grouped_epochs)
        
        # Extract surrogate epochs
        surrogate_data = np.stack([
            raw_data[:, start:start + epoch_duration_samples]
            for start in start_samples
        ])
        
        # Create events array for surrogate epochs
        surrogate_events = np.column_stack([
            np.arange(n_grouped_epochs),  # sample indices (fake, just ordered)
            np.zeros(n_grouped_epochs, dtype=int),  # dummy value (not used)
            np.ones(n_grouped_epochs, dtype=int)    # event ID = 1
        ])
        
        # Build surrogate EpochsArray
        surrogate_epochs = mne.EpochsArray(surrogate_data, raw.info, events=surrogate_events, event_id={'surrogate': 1})
        
        # Optional: confirm shape
        print(surrogate_epochs)
        print(f"✅ Created {len(surrogate_epochs)} surrogate epochs of {epoch_duration_sec:.2f}s each.")
        



        print(f"✅ Final grouped_epochs shape: {grouped_epochs.get_data().shape}")
        print(f"→ Total epochs: {len(grouped_epochs)} (each 11.7s)")
        frontocentral_chs = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6','C3','Cz','C4']

        # Filter channels in grouped_epochs
        available_chs = [ch for ch in frontocentral_chs if ch in grouped_epochs.ch_names]
        grouped_epochs = grouped_epochs.copy().pick_channels(available_chs)
        surrogate_epochs = surrogate_epochs.copy().pick_channels(available_chs)

        
        # === METHOD 1: ITPC ===
        # Define frequencies of interest
        target_freqs = [1/(0.39*3), 1/0.39, 1/(0.39*1.5)]
        
        # Extract data and sampling info
        data = grouped_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        sfreq = grouped_epochs.info['sfreq']
        n_epochs, n_channels, n_times = data.shape
        
        # Perform FFT on each epoch
        fft_data = np.fft.rfft(data, axis=2)  # shape: (n_epochs, n_channels, n_freqs)
        freqs = np.fft.rfftfreq(n_times, d=1/sfreq)
        
        # Compute phase angles at each frequency
        phases = np.angle(fft_data)  # shape: (n_epochs, n_channels, n_freqs)
        
        # Compute ITC = magnitude of mean unit vectors over trials
        itc = np.abs(np.mean(np.exp(1j * phases), axis=0))  # shape: (n_channels, n_freqs)
        
        # Get index of target frequencies
        idx_085 = np.argmin(np.abs(freqs - 1/(0.39*3)))
        idx_256 = np.argmin(np.abs(freqs - 1/0.39))
        idx_170 = np.argmin(np.abs(freqs - 1/(0.39*1.5)))
        
        # Extract ITC at each frequency
        itpc_085 = itc[:, idx_085]
        itpc_256 = itc[:, idx_256]
        itpc_170 = itc[:, idx_170]
        
        # Compute WLI = ITPC(0.85 Hz) / ITPC(2.56 Hz)
        with np.errstate(divide='ignore', invalid='ignore'):
            wli = np.where(itpc_256 != 0, itpc_085 / itpc_256, np.nan)
        
        # Average across channels
        mean_wli_ITPC = np.nanmean(wli)
        # Save to result DataFrame

        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'ITPC_085'] = np.nanmean(itpc_085)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'ITPC_256'] = np.nanmean(itpc_256)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'ITPC_170'] = np.nanmean(itpc_170)
        
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WLI_ITPC'] = mean_wli_ITPC
        #Same calculations for surrogate data 
        # === Surrogate ITPC computation (same as Method 1) ===
        data_surr = surrogate_epochs.get_data()
        fft_data_surr = np.fft.rfft(data_surr, axis=2)
        phases_surr = np.angle(fft_data_surr)
        itc_surr = np.abs(np.mean(np.exp(1j * phases_surr), axis=0))
        
        itpc_085_surr = itc_surr[:, idx_085]
        itpc_256_surr = itc_surr[:, idx_256]
        itpc_170_surr = itc_surr[:, idx_170]
        
        # Save surrogate ITPC values
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'ITPC_085_SURR'] = np.nanmean(itpc_085_surr)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'ITPC_256_SURR'] = np.nanmean(itpc_256_surr)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'ITPC_170_SURR'] = np.nanmean(itpc_170_surr)

        


        

        # === METHOD 2: FFT ===
        evoked = grouped_epochs.average()
        data = evoked.data
        sfreq = evoked.info['sfreq']
        fft_data = np.fft.rfft(data, axis=1)
        freqs = np.fft.rfftfreq(data.shape[1], d=1/sfreq)
        amplitude = np.abs(fft_data)

        idx_085 = np.argmin(np.abs(freqs - 1/(0.39*3)))
        idx_256 = np.argmin(np.abs(freqs -  1/0.39))
        idx_170 = np.argmin(np.abs(freqs - 1/(0.39*1.5)))

        power_at_085 = amplitude[:, idx_085]
        power_at_256 = amplitude[:, idx_256]
        power_at_170 = amplitude[:, idx_170]

        with np.errstate(divide='ignore', invalid='ignore'):
            wli_power = power_at_085 / power_at_256
        wli_power_mean = np.nanmean(wli_power)

        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WLI_POWER'] = wli_power_mean
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'POWER_085'] = np.nanmean(power_at_085)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'POWER_256'] = np.nanmean(power_at_256)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'POWER_170'] = np.nanmean(power_at_170)
        
        # === FFT-based surrogate power ===
        evoked_surr = surrogate_epochs.average()
        data_surr = evoked_surr.data
        fft_data_surr = np.fft.rfft(data_surr, axis=1)
        amplitude_surr = np.abs(fft_data_surr)
        
        power_at_085_surr = amplitude_surr[:, idx_085]
        power_at_256_surr = amplitude_surr[:, idx_256]
        power_at_170_surr = amplitude_surr[:, idx_170]
        
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'POWER_085_SURR'] = np.nanmean(power_at_085_surr)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'POWER_256_SURR'] = np.nanmean(power_at_256_surr)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'POWER_170_SURR'] = np.nanmean(power_at_170_surr)


        print(f"✅ Done: WLI_ITPC = {mean_wli_ITPC:.4f}, WLI_POWER = {wli_power_mean:.4f}")
        
        
        
        # === METHOD 3: Wavelet-based ITPC (Method 4 in Benjamin et al.) ===
        freqs = np.array([1/(0.39*3), 1/0.39, 1/(0.39*1.5)])
        n_cycles = freqs * 5  # standard choice: ~5 cycles per frequency
        
        power_wavelet, itc_wavelet = tfr_morlet(
            grouped_epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=True,
            average=True,  # average across epochs
            decim=1
        )
        
        # Average over time dimension to match Method 4
        mean_itpc_wavelet = itc_wavelet.data.mean(axis=2)  # shape: (n_channels, n_freqs)
        itpc_wavelet_085 = mean_itpc_wavelet[:, 0]
        itpc_wavelet_256 = mean_itpc_wavelet[:, 1]
        itpc_wavelet_170 = mean_itpc_wavelet[:, 2]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            wli_wavelet = np.where(itpc_wavelet_256 != 0, itpc_wavelet_085 / itpc_wavelet_256, np.nan)
        
        mean_wli_wavelet = np.nanmean(wli_wavelet)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WLI_ITPC_WAVELET'] = mean_wli_wavelet
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WAVE_085'] = np.nanmean(itpc_wavelet_085)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WAVE_256'] = np.nanmean(itpc_wavelet_256)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WAVE_170'] = np.nanmean(itpc_wavelet_170)
        # === Wavelet-based surrogate ITPC ===
        power_wavelet_surr, itc_wavelet_surr = tfr_morlet(
            surrogate_epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=True,
            average=True,
            decim=1
        )
        
        mean_itpc_wavelet_surr = itc_wavelet_surr.data.mean(axis=2)
        itpc_wavelet_085_surr = mean_itpc_wavelet_surr[:, 0]
        itpc_wavelet_256_surr = mean_itpc_wavelet_surr[:, 1]
        itpc_wavelet_170_surr = mean_itpc_wavelet_surr[:, 2]
        
        
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WAVE_085_SURR'] = np.nanmean(itpc_wavelet_085_surr)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WAVE_170_SURR'] = np.nanmean(itpc_wavelet_170_surr)
        result_df.loc[(result_df['SUB'] == SUBnr) & (result_df['SESSION'] == SESnr), 'WAVE_256_SURR'] = np.nanmean(itpc_wavelet_256_surr)




    except FileNotFoundError:
        print(f"❌ File not found: {file} or {events_file}")
        result_df.loc[i] = [SUBnr, SESnr, 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND']
        missing_data.append((SUBnr, SESnr))
        continue

    except Exception as e:
        print(f"❌ Error processing Subject {SUBnr}, Session {SESnr}: {e}")
        traceback.print_exc()
        result_df.loc[i] = [SUBnr, SESnr, 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND', 'NOT FOUND']

        continue

print(result_df)
result_df.sort_values(by='SUB', inplace=True)


print("\n📋 Participants with missing or failed EEG data:")
for sub, ses in missing_data:
    print(f"  - Subject: {sub}, Session: {ses}")



#FILTER OUT SUBJECTS WITHOUT DATA AND SAVE A FILE WITH AND WITHOUT R2
 
# Columns to check
wli_cols = ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']

# Create mask for good rows
valid_mask = (
    result_df[wli_cols].notna().all(axis=1) &  # Not NaN
    ~result_df[wli_cols].isin(['NOT FOUND']).any(axis=1)  # Not 'NOT FOUND'
)

# Filtered (valid) rows
filtered_df = result_df[valid_mask].sort_values(by='SUB').reset_index(drop=True)

# Excluded (invalid) rows
excluded_df = result_df[~valid_mask].sort_values(by='SUB').reset_index(drop=True)

# Optional: show just the IDs that were excluded
excluded_ids = excluded_df['SUB'].tolist()

# Output
print("✅ Filtered dataframe (included):")
print(filtered_df)
print(f"\n❌ Excluded {len(excluded_df)} rows (due to missing or 'NOT FOUND'):")
print(excluded_df[['SUB', 'SESSION']])  # You can show more columns if needed
print(f"\n🔎 Excluded subject IDs: {excluded_ids}")

#Save this file as an excel
output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withR2.xlsx"
filtered_df.to_excel(output_path, index=False)
print(f"✅ Saved filtered results to: {output_path}")

filtered_result_df = filtered_df[result_df['SESSION'] != 'R2'].copy()
print(f"📈 Total participants/sessions used (excluding R2): {len(filtered_result_df)}")
output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withoutR2.xlsx"
filtered_result_df.to_excel(output_path, index=False)
print(f"✅ Saved filtered results to: {output_path}")




#Evaluating Neural Entrainment measures on Real and Surrogate EEG Signals


from scipy.stats import ttest_rel
import numpy as np
import pandas as pd

# Define the specific pairs to compare
pairs = [
    ('ITPC_085', 'ITPC_085_SURR'),
    ('ITPC_256', 'ITPC_256_SURR')
]

# Ensure numeric
for col1, col2 in pairs:
    filtered_result_df[[col1, col2]] = filtered_result_df[[col1, col2]].apply(pd.to_numeric, errors='coerce')

# Store results
results = []

# Run tests
for col1, col2 in pairs:
    df_clean = filtered_result_df[[col1, col2]].dropna()
    if len(df_clean) < 2:
        print(f"⚠️ Not enough data for {col1} vs {col2}. Skipping.")
        results.append((col1, col2, np.nan, np.nan, np.nan))
        continue

    t, p = ttest_rel(df_clean[col1], df_clean[col2])
    one_tailed_p = p / 2 if t > 0 else 1 - (p / 2)
    results.append((col1, col2, t, one_tailed_p, len(df_clean)))

# Print raw results
print("\n📊 One-tailed paired t-tests:")
for col1, col2, t, p, n in results:
    print(f"{col1} vs {col2}: t = {t:.3f}, p = {p:.4f}, n = {n}")

# Apply Bonferroni correction
raw_pvals = [r[3] for r in results]
corrected_pvals = [min(p * len(pairs), 1.0) if not np.isnan(p) else np.nan for p in raw_pvals]

print("\n🔍 Bonferroni-corrected p-values:")
for (col1, col2, _, p, _), corrected in zip(results, corrected_pvals):
    print(f"{col1} vs {col2}: corrected p = {corrected:.4f}")




#Evidence of Statistical Learning Using the SICR Task
import pandas as pd
from scipy.stats import ttest_rel

# Load the Excel file
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\SICR DATA\SICR SCORED2\DL_scores.xlsx"
df = pd.read_excel(file_path)

# Define the columns to compare
comparisons = {
    "Overall": ("Mean DL similarity - targets (overall)", "Mean DL similarity - foils (overall)"),
    "Short": ("Mean DL similarity - targets (short)", "Mean DL similarity - foils (short)"),
    "Long": ("Mean DL similarity - targets (long)", "Mean DL similarity - foils (long)")
}

# Run one-tailed paired t-tests
for condition, (target_col, foil_col) in comparisons.items():
    # Drop missing values
    data = df[[target_col, foil_col]].dropna()
    target = data[target_col]
    foil = data[foil_col]

    # Perform two-tailed paired t-test
    t_stat, p_two_tailed = ttest_rel(target, foil)

    # Adjust one-tailed direction based on condition
    if condition == "Short":
        # Test whether Foils > Targets
        if t_stat < 0:
            p_one_tailed = p_two_tailed / 2
        else:
            p_one_tailed = 1 - (p_two_tailed / 2)
    else:
        # Test whether Targets > Foils
        if t_stat > 0:
            p_one_tailed = p_two_tailed / 2
        else:
            p_one_tailed = 1 - (p_two_tailed / 2)

    # Print results
    print(f"\n--- {condition} ---")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (one-tailed): {p_one_tailed:.6f}")
    print(f"Mean Target: {target.mean():.4f}")
    print(f"Mean Foil:   {foil.mean():.4f}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np

# Load data
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\SICR DATA\SICR SCORED2\DL_scores.xlsx"
df = pd.read_excel(file_path)

# Define condition-to-column mapping
comparisons = {
    "Overall": ("Mean DL similarity - targets (overall)", "Mean DL similarity - foils (overall)"),
    "3-syllable": ("Mean DL similarity - targets (short)", "Mean DL similarity - foils (short)"),
    "6-syllable": ("Mean DL similarity - targets (long)", "Mean DL similarity - foils (long)")
}

# Convert to long format
long_data = []
for condition, (target_col, foil_col) in comparisons.items():
    temp_df = df[[target_col, foil_col]].dropna()
    for i, row in temp_df.iterrows():
        long_data.append({'Participant': i, 'Condition': condition, 'Type': 'Target', 'Score': row[target_col]})
        long_data.append({'Participant': i, 'Condition': condition, 'Type': 'Foil', 'Score': row[foil_col]})

plot_df = pd.DataFrame(long_data)

# === Compute means and SEs ===
summary_df = (
    plot_df.groupby(['Condition', 'Type'])['Score']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
summary_df['se'] = summary_df['std'] / np.sqrt(summary_df['count'])

# === Plot ===
plt.figure(figsize=(10, 6))

# Plot individual scores (light color)
sns.stripplot(data=plot_df, x='Condition', y='Score', hue='Type',
              dodge=True, jitter=0.2, alpha=0.4,
              palette={'Target': 'steelblue', 'Foil': 'coral'})

# Plot group means with SE error bars
for i, row in summary_df.iterrows():
    x_pos = list(comparisons.keys()).index(row['Condition'])  # base x
    x_offset = -0.2 if row['Type'] == 'Target' else 0.2
    x = x_pos + x_offset
    color = 'navy' if row['Type'] == 'Target' else 'darkred'
    plt.errorbar(x, row['mean'], yerr=row['se'], fmt='D', color=color,
                 capsize=5, markersize=8, elinewidth=1.5)

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Individual Target Scores',
           markerfacecolor='steelblue', markersize=8, alpha=0.4),
    Line2D([0], [0], marker='o', color='w', label='Individual Foil Scores',
           markerfacecolor='coral', markersize=8, alpha=0.4),
    Line2D([0], [0], marker='D', color='navy', label='Mean Target ± SE', markersize=8),
    Line2D([0], [0], marker='D', color='darkred', label='Mean Foil ± SE', markersize=8)
]

# Final formatting
plt.title("SICR Similarity Scores by Condition and Type")
plt.ylabel("Mean Similarity Score")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.legend(handles=legend_elements, title="Sequence Type")
plt.tight_layout()
plt.show()







#Relationship Between Methods of Calculating Online Statistical Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, t
from itertools import combinations

# === Load the data ===
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withoutR2.xlsx"
df = pd.read_excel(file_path)

# === Select the relevant WLI columns ===
wli_cols = ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']

# === Define display names for axis labels ===
display_names = {
    'WLI_ITPC': 'FFT-based ITPC',
    'WLI_POWER': 'Evoked Power Spectrum',
    'WLI_ITPC_WAVELET': 'Wavelet-based ITPC'
}

# === Function to calculate 95% confidence interval for Pearson r ===
def pearson_ci(r, n, alpha=0.05):
    if n < 4:
        return (np.nan, np.nan)
    stderr = 1.0 / np.sqrt(n - 3)
    delta = t.ppf(1 - alpha / 2, n - 2) * stderr
    fisher_z = np.arctanh(r)
    ci_lower = np.tanh(fisher_z - delta)
    ci_upper = np.tanh(fisher_z + delta)
    return ci_lower, ci_upper

# === Compute pairwise correlations ===
results = []
for col1, col2 in combinations(wli_cols, 2):
    paired_data = df[[col1, col2]].dropna()
    if len(paired_data) >= 3:
        r, p = pearsonr(paired_data[col1], paired_data[col2])
        ci_low, ci_high = pearson_ci(r, len(paired_data))
        results.append({
            'Variable 1': display_names[col1],
            'Variable 2': display_names[col2],
            'Pearson r': r,
            'p-value': p,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high,
            'N': len(paired_data)
        })

# === Print correlation table ===
correlation_df = pd.DataFrame(results)
print("\n=== Pairwise Correlations ===")
print(correlation_df)

# === Plot each pair with readable labels ===
pairs = list(combinations(wli_cols, 2))

for x_var, y_var in pairs:
    data = df[[x_var, y_var]].dropna()
    if len(data) >= 3:
        r, p = pearsonr(data[x_var], data[y_var])
        plt.figure(figsize=(6, 4))
        sns.regplot(x=x_var, y=y_var, data=data, scatter_kws={'alpha': 0.6})
        plt.title(f"{display_names[x_var]} vs {display_names[y_var]}\nPearson r = {r:.2f}, p = {p:.4f}")
        plt.xlabel(display_names[x_var])
        plt.ylabel(display_names[y_var])
        plt.tight_layout()
        plt.show()























#Correlational Analysis of WLI and SICR Task

import pandas as pd
import numpy as np
from pingouin import partial_corr

# === File paths ===
wli_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withR2.xlsx"
sicr_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\SICR DATA\SICR SCORED\DL_scores.xlsx"

# === Load WLI and SICR data ===
wli_df = pd.read_excel(wli_path, usecols=['SUB', 'SESSION', 'WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET'])
sicr_df = pd.read_excel(sicr_path)

# === Extract SUB and SESSION from 'File name' ===
sicr_df['SUB'] = sicr_df['File name'].str.extract(r'_SUB(\d+)_')[0].astype(int)
sicr_df['SESSION'] = sicr_df['File name'].str.extract(r'_SUB\d+_(L\d)_')[0]

# === Select required SICR columns ===
sicr_clean = sicr_df[['SUB', 'SESSION', 'Learning score (overall)', 'Mean DL similarity - foils (overall)']]

# === Log-transform WLI scores ===
for col in ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']:
    wli_df[f'log_{col}'] = np.log(wli_df[col])

# === Merge datasets ===
merged_df = pd.merge(
    wli_df,
    sicr_clean,
    on=['SUB', 'SESSION'],
    how='inner'
)

# === Run partial correlations with manual t and df ===
for log_col in ['log_WLI_ITPC', 'log_WLI_POWER', 'log_WLI_ITPC_WAVELET']:
    result = partial_corr(
        data=merged_df,
        x=log_col,
        y='Learning score (overall)',
        covar='Mean DL similarity - foils (overall)',
        method='pearson'
    )
    r = result['r'].values[0]
    n = result['n'].values[0]
    dfree = n - 2
    t_stat = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2)

    print(f"\nPartial correlation for {log_col} and learning effect (controlling for foil recall):")
    print(f"r = {r:.3f}, t({dfree}) = {t_stat:.2f}, p = {result['p-val'].values[0]:.4f}, 95% CI = {result['CI95%'].values[0]}")











#TEST RETEST RELIABILITY PILOT
import pandas as pd

# Load the Excel file
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withR2.xlsx"
df = pd.read_excel(file_path)

# Filter for only R1 and R2 sessions
filtered_df = df[df['SESSION'].isin(['R1', 'R2'])].copy()

# Optional: check how many rows were retained
print(f"✅ Total R1 and R2 sessions: {filtered_df.shape[0]}")

# If you want to do something with the filtered data:
print(filtered_df.head())

# Group by participant and count how many sessions they have
session_counts = filtered_df.groupby('SUB')['SESSION'].nunique()

# Find participants who have both R1 and R2
complete_participants = session_counts[session_counts == 2]

# Filter the dataframe to only include those with both R1 and R2
df_complete_pairs = filtered_df[filtered_df['SUB'].isin(complete_participants.index)]

# Print how many participants had both R1 and R2
print(f"✅ Total participants with both R1 and R2 sessions: {len(complete_participants)}")

# Optional: show the list
print(complete_participants.index.tolist())


import pandas as pd
from scipy.stats import pearsonr

# Load the Excel file
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withR2.xlsx"
df = pd.read_excel(file_path)

# Filter for only R1 and R2 sessions
df = df[df['SESSION'].isin(['R1', 'R2'])].copy()

# Pivot to wide format: one row per participant, separate R1 and R2 columns
pivot_df = df.pivot(index='SUB', columns='SESSION', values=['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET'])

# Drop participants missing either R1 or R2
pivot_df = pivot_df.dropna()

# Compute correlations
methods = ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']

print("📈 Correlation between R1 and R2 for each WLI method:")
for method in methods:
    r1 = pivot_df[(method, 'R1')]
    r2 = pivot_df[(method, 'R2')]
    r, p = pearsonr(r1, r2)
    print(f"{method}: r = {r:.3f}, p = {p:.4f}, n = {len(r1)}")



















#Participant numbers

import pandas as pd

# Load the Excel file
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withR2.xlsx"  # Update this path if needed
file_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withoutR2.xlsx"
df = pd.read_excel(file_path)

# Count the occurrences of each session type
session_counts = df['SESSION'].value_counts()

# Calculate the total number of sessions
total_sessions = session_counts.sum()

# Display the results
print("Session Counts:")
print(session_counts)
print("\nTotal Sessions:", total_sessions)







