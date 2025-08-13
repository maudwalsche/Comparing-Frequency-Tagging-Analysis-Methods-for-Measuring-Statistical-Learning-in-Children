# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 13:55:48 2025

@author: User
"""


#Data Analyses of MP2 Maurice De Walsche
#The raw data can't be shared in this MP2, but for all RQ's to be answered the subsequent excel file with WLI and SICR calculations will be available for analyses to be checked
#The right path file just needs to be added of the excel files that are present in the github



#Calculating WLI of the three different methods from the raw data
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
file_path = r"C:\Users\User\OneDrive - UGent\TULIP - Towards Understanding Learning In Preterms - Data (No personal identifiers!)\EEG\EEG Logbook.xlsx"
df = pd.read_excel(file_path, header=3)

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
mean_epochs_all_subjects = []

# === Loop over subject-session pairs ===
for i, (subject, session) in enumerate(subject_session_array):
    SUBnr = str(subject).strip()
    SESnr = str(session).strip().upper()
    result_df.loc[i, 'SUB'] = SUBnr
    result_df.loc[i, 'SESSION'] = SESnr

    print(f"\n📌 Subject: {SUBnr}, Session: '{SESnr}'")

    file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_preproc_eeg.fif"
    events_file = f"TULIP_SUB{SUBnr}_{SESnr}_EEG_events.npy"

    working_dir = r"C:\Users\User\OneDrive - UGent\TULIP - Towards Understanding Learning In Preterms - Data (No personal identifiers!)\EEG\Processed"
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
                # Count epochs in each block
        n_block1 = len(block1)
        n_block2 = len(block2)
        n_block3 = len(block3)
        
        # Compute mean for this subject
        mean_epochs_per_block = np.mean([n_block1, n_block2, n_block3])
        mean_epochs_all_subjects.append(mean_epochs_per_block)



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
        # === Generate surrogate epochs ===
        epoch_duration_sec = 11.7
        epoch_samples = int(epoch_duration_sec * sfreq)
        raw_data = cleaned_raw.get_data()

        n_surrogates = 10
        surrogate_data_list = []
        
        # Generate 10 surrogate datasets
        for k in range(n_surrogates):
            rng = np.random.default_rng(seed=42 + k)  # different seed for each surrogate
            start_samples = rng.integers(
                0, raw_data.shape[1] - epoch_samples,
                size=len(grouped_epochs)
            )
            surrogate_data = np.stack([
                raw_data[:, s:s + epoch_samples] for s in start_samples
            ])
            surrogate_data_list.append(surrogate_data)
        
        # Convert list to array: shape = (n_surrogates, n_epochs, n_channels, epoch_samples)
        surrogate_data_array = np.array(surrogate_data_list)
        
        # Average across surrogates: shape = (n_epochs, n_channels, epoch_samples)
        mean_surrogate_data = np.mean(surrogate_data_array, axis=0)
        
        # Create a single EpochsArray from the averaged surrogate data
        surrogate_epochs = mne.EpochsArray(mean_surrogate_data, raw.info)
        
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

if mean_epochs_all_subjects:
    overall_mean = np.mean(mean_epochs_all_subjects)
    print(f"\n📊 Overall mean number of epochs per block across participants: {overall_mean:.2f}")
else:
    print("\n⚠️ No valid participants processed to compute mean epochs per block.")


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
print(len(filtered_df['SUB'].unique()))
print(f"\n❌ Excluded {len(excluded_df)} rows (due to missing or 'NOT FOUND'):")
print(excluded_df[['SUB', 'SESSION']])  # You can show more columns if needed
print(f"\n🔎 Excluded subject IDs: {excluded_ids}")

#Save this file as an excel
output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withR2.xlsx"
filtered_df.to_excel(output_path, index=False)
print(f"✅ Saved filtered results to: {output_path}")

filtered_result_df = filtered_df[filtered_df['SESSION'] != 'R2'].copy()
print(f"📈 Total participants/sessions used (excluding R2): {len(filtered_result_df)}")
output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\filtered_results_withoutR2.xlsx"
# Remove subject 9220, session L2

filtered_result_df.to_excel(output_path, index=False)
print(f"✅ Saved filtered results to: {output_path}")

len(filtered_result_df)
unique_pairs = filtered_result_df[['SUB', 'SESSION']].drop_duplicates()
print(unique_pairs)







#information about the sample, add correct path file for the subsequent Excel file, participant metadata cant be shared.
import pandas as pd

# --- File paths ---
filtered_path = r"filtered_results_withoutR2.xlsx"
metadata_path = r"C:\Users\User\OneDrive - UGent\TULIP - Towards Understanding Learning In Preterms - Data (No personal identifiers!)\participant_metadata.csv"

# --- Load filtered results ---
filtered_df = pd.read_excel(filtered_path)

# Find column that contains 'sub' (case-insensitive)
sub_col_filtered = next((c for c in filtered_df.columns if 'sub' in c.lower()), None)
if sub_col_filtered is None:
    raise ValueError("No column containing 'sub' found in filtered results file.")

sub_numbers = pd.Series(filtered_df[sub_col_filtered].unique())

# --- Load metadata ---
meta_df = pd.read_csv(metadata_path)

# Identify subject column in metadata
sub_col_meta = next((c for c in meta_df.columns if 'sub' in c.lower()), None)
if sub_col_meta is None:
    raise ValueError("No column containing 'sub' found in participant metadata file.")

# --- Filter metadata to subjects of interest ---
filtered_metadata = meta_df[meta_df[sub_col_meta].isin(sub_numbers)]

# Keep relevant columns
keep_cols = [sub_col_meta, "sex", "age", "prematurity_cat", "ga_weeks"]
filtered_metadata = filtered_metadata[keep_cols].copy()

# --- Clean sex column ---
filtered_metadata["sex"] = (
    filtered_metadata["sex"]
    .astype(str)
    .str.strip()
    .str.capitalize()
    .replace({"M": "Boy", "F": "Girl"})
)

# --- Stats ---
sex_counts = filtered_metadata["sex"].value_counts()
prematurity_counts = filtered_metadata["prematurity_cat"].value_counts()
mean_age = filtered_metadata["age"].mean()
sd_age = filtered_metadata["age"].std()
mean_ga = filtered_metadata["ga_weeks"].mean()
sd_ga = filtered_metadata["ga_weeks"].std()

# --- Output ---
print("Filtered Metadata:")
print(filtered_metadata)

print("\nNumber of Girls and Boys:")
print(sex_counts)

print("\nNumber of each prematurity category:")
print(prematurity_counts)

print(f"\nMean age: {mean_age:.3f}")
print(f"SD of age: {sd_age:.3f}")

print(f"\nMean GA weeks: {mean_ga:.3f}")
print(f"SD of GA weeks: {sd_ga:.3f}")





# Evidence of Statistical Learning Frequency tagging, add pathfile of subsequent excel file
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

filtered_path = r"filtered_results_withoutR2.xlsx"
filtered_result_df = pd.read_excel(filtered_path)

# All frequency pairings across different metrics
pairs = [
    ('ITPC_085', 'ITPC_085_SURR'),
    ('ITPC_256', 'ITPC_256_SURR'),
    ('POWER_085', 'POWER_085_SURR'),
    ('POWER_256', 'POWER_256_SURR'),
    ('WAVE_085', 'WAVE_085_SURR'),
    ('WAVE_256', 'WAVE_256_SURR')
]

# Ensure numeric values
for col1, col2 in pairs:
    filtered_result_df[[col1, col2]] = filtered_result_df[[col1, col2]].apply(pd.to_numeric, errors='coerce')

# Run tests and collect results
results = []
for col1, col2 in pairs:
    df_clean = filtered_result_df[[col1, col2]].dropna()
    if len(df_clean) < 2:
        print(f"⚠️ Not enough data for {col1} vs {col2}. Skipping.")
        results.append((col1, col2, np.nan, np.nan, np.nan))
        continue

    # Two-tailed paired t-test
    t, p = ttest_rel(df_clean[col1], df_clean[col2])
    results.append((col1, col2, t, p, len(df_clean)))

# === Report ===
print("\n📊 Two-tailed paired t-tests:")
for col1, col2, t, p, n in results:
    print(f"{col1} vs {col2}: t = {t:.3f}, p = {p:.4f}, n = {n}")

# Bonferroni correction for 6 comparisons
bonferroni_n = 6
raw_pvals = [r[3] for r in results]
corrected_pvals = [min(p * bonferroni_n, 1.0) if not np.isnan(p) else np.nan for p in raw_pvals]

print("\n🔍 Bonferroni-corrected p-values (n=6):")
for (col1, col2, _, p, _), corrected in zip(results, corrected_pvals):
    print(f"{col1} vs {col2}: corrected p = {corrected:.4f}")





# Evidence of Statistical Learning Using the SICR Task
import pandas as pd
from scipy.stats import ttest_rel

# Load the Excel file
file_path = r"DL_scores.xlsx"
df = pd.read_excel(file_path)

# Filter out rows where 'File name' contains 'R2'
df = df[~df['File name'].astype(str).str.contains("R2", case=False, na=False)]

# Optional: reset index
df = df.reset_index(drop=True)

# Define the columns to compare
comparisons = {
    "Overall": ("Mean DL similarity - targets (overall)", "Mean DL similarity - foils (overall)"),
    "Short": ("Mean DL similarity - targets (short)", "Mean DL similarity - foils (short)"),
    "Long": ("Mean DL similarity - targets (long)", "Mean DL similarity - foils (long)")
}

# Run two-tailed paired t-tests
for condition, (target_col, foil_col) in comparisons.items():
    # Drop missing values
    data = df[[target_col, foil_col]].dropna()
    target = data[target_col]
    foil = data[foil_col]

    n = len(data)  # sample size

    # Perform two-tailed paired t-test
    t_stat, p_two_tailed = ttest_rel(target, foil)

    # Print results
    print(f"\n--- {condition} ---")
    print(f"Sample size (n): {n}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value (two-tailed): {p_two_tailed:.6f}")
    print(f"Mean Target: {target.mean():.4f}")
    print(f"Mean Foil:   {foil.mean():.4f}")

    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np

# Load data
file_path = r"DL_scores.xlsx"
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
plt.xlabel("")
plt.title("SICR  Scores by Target and Foils")
plt.ylabel("Mean SICR Score")
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)
plt.legend(handles=legend_elements, title="Sequence Type")
plt.tight_layout()
plt.show()





#Correlations between different EEG measures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, t
from itertools import combinations

# === Load the data ===
file_path = r"filtered_results_withoutR2.xlsx"
df = pd.read_excel(file_path)

# === Select the relevant WLI columns ===
wli_cols = ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']

# === Define display names for axis labels ===
display_names = {
    'WLI_ITPC': 'FFT-based ITPC',
    'WLI_POWER': 'Evoked Power Spectrum',
    'WLI_ITPC_WAVELET': 'Wavelet-based ITPC'
}

# === Apply log transformation to WLI columns (using natural log) ===
log_df = df.copy()
for col in wli_cols:
    # To avoid issues with log(0), we use np.log1p for log(1 + x), or alternatively filter out non-positive values
    # Uncomment one of the two options below

    # Option 1: Use np.log1p to handle 0 safely
    # log_df[col] = np.log1p(log_df[col])

    # Option 2: Use natural log and drop/NaN non-positive values
    log_df[col] = np.where(log_df[col] > 0, np.log(log_df[col]), np.nan)

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

# === Compute pairwise correlations on log-transformed data ===
results = []
for col1, col2 in combinations(wli_cols, 2):
    paired_data = log_df[[col1, col2]].dropna()
    if len(paired_data) >= 3:
        r, p = pearsonr(paired_data[col1], paired_data[col2])
        ci_low, ci_high = pearson_ci(r, len(paired_data))
        results.append({
            'Variable 1': display_names[col1],
            'Variable 2': display_names[col2],
            'Pearson r': r,
            't-value': t_val,
            'p-value': p,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high,
            'N': len(paired_data)
        })

# === Print correlation table ===
correlation_df = pd.DataFrame(results)
print("\n=== Pairwise Correlations (Log-Transformed) ===")
print(correlation_df)
# === Apply Bonferroni correction ===
n_tests = len(correlation_df)
correlation_df['p-value (Bonferroni)'] = (correlation_df['p-value'] * n_tests).clip(upper=1.0)
correlation_df['Significant (Bonferroni)'] = correlation_df['p-value (Bonferroni)'] < 0.05

# after building correlation_df
import numpy as np
from scipy.stats import t as student_t

# t for each correlation:  t = r * sqrt(n-2) / sqrt(1 - r^2)
correlation_df['t_value'] = correlation_df.apply(
    lambda row: row['Pearson r'] * np.sqrt(row['N'] - 2) / np.sqrt(1 - row['Pearson r']**2),
    axis=1
)

# Optional: verify p from t (pearsonr gives two-tailed p by default)
correlation_df['p_from_t'] = 2 * student_t.sf(np.abs(correlation_df['t_value']),
                                              df=correlation_df['N'] - 2)

# Show more precision so small differences don’t look identical
import pandas as pd
pd.set_option('display.float_format', '{:.6f}'.format)
print(correlation_df[['Variable 1','Variable 2','Pearson r','N','t_value','p-value','p_from_t']])
 
# === Print updated correlation table ===
print("\n=== Pairwise Correlations with Bonferroni Correction ===")
print(correlation_df)

# === Create subplots ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Only use three subplots: (a), (b), and (c)
subplot_labels = ['(a)', '(b)', '(c)']
pairs = list(combinations(wli_cols, 2))

for i, (x_var, y_var) in enumerate(pairs):
    data = log_df[[x_var, y_var]].dropna()
    if len(data) >= 3:
        r, p = pearsonr(data[x_var], data[y_var])
        ax = axes[i]
        sns.regplot(x=x_var, y=y_var, data=data, scatter_kws={'alpha': 0.6}, ax=ax)

        # Axis labels
        ax.set_xlabel(f"{display_names[x_var]} WLI")
        ax.set_ylabel(f"{display_names[y_var]} WLI")

        # Format p-value
        p_str = "< 0.001" if p < 0.001 else f"= {p:.4f}"

        # Title with r and formatted p
        ax.set_title(f"{subplot_labels[i]}  r = {r:.2f}, p {p_str}", loc='center', fontsize=11)

# Remove the unused 4th subplot
fig.delaxes(axes[3])

# Main title
fig.suptitle("Correlations between Frequency Tagging Methods", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



















#Correlational Analysis of WLI and SICR Task

import pandas as pd
import numpy as np
from pingouin import partial_corr

# === File paths ===
wli_path = r"filtered_results_withoutR2.xlsx"
sicr_path = r"DL_scores.xlsx"

# === Load WLI and SICR data ===
wli_df = pd.read_excel(wli_path, usecols=['SUB', 'SESSION', 'WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET'])
sicr_df = pd.read_excel(sicr_path)

# === Extract SUB and SESSION from 'File name' ===
sicr_df['SUB'] = sicr_df['File name'].str.extract(r'_SUB(\d+)_')[0].astype(int)
sicr_df['SESSION'] = sicr_df['File name'].str.extract(r'_SUB\d+_([LR]\d)_')[0]

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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec

display_names = {
    'log_WLI_ITPC': 'FFT-based ITPC (log)',
    'log_WLI_POWER': 'Power method (log)',  # renamed label
    'log_WLI_ITPC_WAVELET': 'Wavelet-based ITPC (log)'
}

panel_labels = ['(a)', '(b)', '(c)']
# Order: (a) Power method, (b) FFT-based ITPC, (c) Wavelet-based ITPC
columns = ['log_WLI_POWER', 'log_WLI_ITPC', 'log_WLI_ITPC_WAVELET']
panel_labels = ['(a)', '(b)', '(c)']  # keeps labels aligned with the new order

# === Create custom layout ===
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2)

# Define positions
axs = [
    fig.add_subplot(gs[0, 0]),  # (a) top-left
    fig.add_subplot(gs[0, 1]),  # (b) top-right
    fig.add_subplot(gs[1, 0])   # (c) bottom-left (centered visually)
]

# Hide the bottom-right subplot space to center the third panel
gs.update(wspace=0.3, hspace=0.4)
fig.suptitle("Partial Correlations: WLI vs SICR Learning Effect (Controlling for Foil Recall)", fontsize=15)

# === Loop through plots ===
for i, (log_col, label) in enumerate(zip(columns, panel_labels)):
    ax = axs[i]

    # Clean and residualize
    temp_df = merged_df[[log_col, 'Learning score (overall)', 'Mean DL similarity - foils (overall)']].dropna()
    X_control = temp_df[['Mean DL similarity - foils (overall)']].values
    y1 = temp_df[log_col].values
    y2 = temp_df['Learning score (overall)'].values

    model1 = LinearRegression().fit(X_control, y1)
    model2 = LinearRegression().fit(X_control, y2)
    residuals_x = y1 - model1.predict(X_control)
    residuals_y = y2 - model2.predict(X_control)

    # Correlation on residuals
    r, p = pearsonr(residuals_x, residuals_y)

    # Plot
    sns.regplot(x=residuals_x, y=residuals_y, ax=ax, scatter_kws={'alpha': 0.6})
    ax.set_xlabel(display_names[log_col] + "\nWLI (res)", fontsize=10)
    if i == 0 or i == 2:
        ax.set_ylabel("SICR Learning Effect (res)", fontsize=10)
    else:
        ax.set_ylabel("")
    ax.set_title(f"r = {r:.2f}, p = {p:.4f}", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel label
    ax.text(-0.1, 1.1, label, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

# Remove the empty bottom-right panel space
fig.add_subplot(gs[1, 1]).axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




























#TEST RETEST RELIABILITY PILOT
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

# Load Excel
file_path = r"filtered_results_withR2.xlsx"
df = pd.read_excel(file_path)

# Keep only subjects that have an R2 session
subjects_with_r2 = df.loc[df['SESSION'] == 'R2', 'SUB'].unique()

# Keep only R2/R1/L2 rows for those subjects
subset = df[
    df['SUB'].isin(subjects_with_r2) &
    df['SESSION'].isin(['R2', 'R1', 'L2'])
].copy()

# Pick earliest session per subject (R1 preferred over L2)
priority = {'R1': 0, 'L2': 1}
other_df = (
    subset[subset['SESSION'].isin(['R1', 'L2'])]
      .assign(_prio=lambda d: d['SESSION'].map(priority))
      .sort_values(['SUB', '_prio'])
      .groupby('SUB', as_index=False)
      .first()
      .drop(columns=['_prio'])
)

# Get R2 rows
r2_df = subset[subset['SESSION'] == 'R2'].copy()

# Merge into wide table
paired = r2_df.merge(
    other_df,
    on='SUB',
    how='left',
    suffixes=('_R2', '_OTHER')  # _OTHER = earlier session
)

# Keep only columns of interest
pivot_df = paired[
    ['SUB',
     'WLI_ITPC_R2', 'WLI_POWER_R2', 'WLI_ITPC_WAVELET_R2',
     'WLI_ITPC_OTHER', 'WLI_POWER_OTHER', 'WLI_ITPC_WAVELET_OTHER']
].copy()

# Calculate Pearson r, p, t, df
correlations = {}
for measure in ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']:
    time1 = f"{measure}_OTHER"  # first session
    time2 = f"{measure}_R2"     # second session

    valid_data = pivot_df[[time1, time2]].dropna()
    n = len(valid_data)
    
    if n > 2:  # need at least 3 points for correlation
        r_value, p_value = pearsonr(valid_data[time1], valid_data[time2])
        t_val = (r_value * np.sqrt(n - 2)) / np.sqrt(1 - r_value**2) if abs(r_value) < 1 else np.inf
    else:
        r_value, p_value, t_val = np.nan, np.nan, np.nan

    correlations[measure] = {
        'Pearson_r': r_value,
        't_stat': t_val,
        'df': n - 2,
        'p_value': p_value,
        'n_pairs': n
    }

# Make results table
corr_df = pd.DataFrame(correlations).T
print(corr_df)


#Plot with outliers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec

# === Use your already-prepared pivot_df from earlier steps ===
# pivot_df contains SUB, *_R2, *_OTHER columns
# Example:
# pivot_df = paired[['SUB',
#     'WLI_ITPC_R2', 'WLI_POWER_R2', 'WLI_ITPC_WAVELET_R2',
#     'WLI_ITPC_OTHER', 'WLI_POWER_OTHER', 'WLI_ITPC_WAVELET_OTHER']].copy()

# --- Setup ---
methods = ['WLI_POWER', 'WLI_ITPC', 'WLI_ITPC_WAVELET']
panel_labels = ['(a)', '(b)', '(c)']
labels = {
    'WLI_ITPC': 'FFT-based ITPC',
    'WLI_POWER': 'Evoked Power Spectrum',
    'WLI_ITPC_WAVELET': 'Wavelet-based ITPC'
}

# Create 2x2 subplot layout
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0])
]
fig.add_subplot(gs[1, 1]).axis('off')  # Empty bottom-right

fig.suptitle("Test–Retest Reliability of WLI Measures (First vs Second Session)", fontsize=15)

# Loop through plots
for i, method in enumerate(methods):
    ax = axs[i]
    x = pivot_df[f"{method}_OTHER"]  # First session (R1 or L2)
    y = pivot_df[f"{method}_R2"]     # Second session (R2)

    # Drop NaNs for correlation
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    x, y = valid['x'], valid['y']

    # Correlation stats
    r, p = pearsonr(x, y)
    n = len(x)
    dfree = n - 2
    t_val = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2) if r**2 < 1 else np.inf

    # Plot
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.7})
    ax.set_xlabel(f"{labels[method]} (First Session)", fontsize=10)
    ax.set_ylabel(f"{labels[method]} (Second Session)", fontsize=10)
    ax.set_title(f"r({dfree}) = {r:.2f}, t = {t_val:.2f}, p = {p:.3f}", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel label
    ax.text(-0.1, 1.1, panel_labels[i],
            transform=ax.transAxes,
            fontsize=13, fontweight='bold',
            va='top', ha='right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec

# Load the Excel file
file_path = r"filtered_results_withR2.xlsx"
df = pd.read_excel(file_path)

# Filter for only R1 and R2 sessions
df = df[df['SESSION'].isin(['R1', 'R2'])].copy()

# Pivot: one row per participant
pivot_df = df.pivot(index='SUB', columns='SESSION', values=['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET'])
pivot_df = pivot_df.dropna()

# Setup
# Order for panels: (a) Power, (b) FFT-based ITPC, (c) Wavelet-based ITPC
methods = ['WLI_POWER', 'WLI_ITPC', 'WLI_ITPC_WAVELET']
panel_labels = ['(a)', '(b)', '(c)']
labels = {
    'WLI_ITPC': 'FFT-based ITPC',
    'WLI_POWER': 'Evoked Power Spectrum',
    'WLI_ITPC_WAVELET': 'Wavelet-based ITPC'
}


# Create 2x2 subplot layout
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0])
]
fig.add_subplot(gs[1, 1]).axis('off')  # Empty bottom-right

fig.suptitle("Test–Retest Reliability of WLI Measures (R1 vs R2)", fontsize=15)

# Loop through plots
for i, method in enumerate(methods):
    ax = axs[i]
    x = pivot_df[(method, 'R1')]
    y = pivot_df[(method, 'R2')]

    # Correlation stats
    r, p = pearsonr(x, y)
    n = len(x)
    dfree = n - 2
    t_val = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2) if r**2 < 1 else np.inf

    # Plot
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.7})
    ax.set_xlabel(f"{labels[method]} (R1)", fontsize=10)
    ax.set_ylabel(f"{labels[method]} (R2)", fontsize=10)
    ax.set_title(f"r({dfree}) = {r:.2f}, t = {t_val:.2f}, p = {p:.3f}", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel label
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#With outliers removed

import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Load the Excel file
file_path = r"filtered_results_withR2.xlsx"
df = pd.read_excel(file_path)

# Filter for only R1 and R2 sessions
df = df[df['SESSION'].isin(['R1', 'R2'])].copy()

# Pivot to wide format: one row per participant, separate R1 and R2 columns
pivot_df = df.pivot(index='SUB', columns='SESSION', values=['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET'])

# Drop participants missing either R1 or R2
# Drop participants missing either R1 or R2
pivot_df = pivot_df.dropna()

# Remove outliers for each method
methods = ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']

for method in methods:
    r1 = pivot_df[(method, 'R1')]
    r2 = pivot_df[(method, 'R2')]

    # Mean and SD for both sessions combined
    combined = pd.concat([r1, r2])
    mean_val = combined.mean()
    std_val = combined.std()

    # Keep only those within ±3 SD of the mean
    mask = (r1.between(mean_val - 3*std_val, mean_val + 3*std_val) &
            r2.between(mean_val - 3*std_val, mean_val + 3*std_val))
    pivot_df = pivot_df[mask]

# Now pivot_df has outliers removed


print("📈 Correlation between R1 and R2 for each WLI method:")
for method in methods:
    r1 = pivot_df[(method, 'R1')]
    r2 = pivot_df[(method, 'R2')]
    r, p = pearsonr(r1, r2)
    n = len(r1)
    dfree = n - 2
    if r**2 < 1:  # Avoid division by zero
        t_val = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2)
    else:
        t_val = np.inf  # Perfect correlation (rare)

    print(f"{method}: r({dfree}) = {r:.3f}, t = {t_val:.2f}, p = {p:.4f}, n = {n}")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# === Order for plotting: (a) Power, (b) FFT-based ITPC, (c) Wavelet-based ITPC ===
methods = ['WLI_POWER', 'WLI_ITPC', 'WLI_ITPC_WAVELET']
labels = {
    'WLI_ITPC': 'FFT-based ITPC',
    'WLI_POWER': 'Evoked Power Spectrum',
    'WLI_ITPC_WAVELET': 'Wavelet-based ITPC'
}
panel_labels = ['(a)', '(b)', '(c)']

# === Create figure layout ===
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2)
axs = [
    fig.add_subplot(gs[0, 0]),  # (a) Power method
    fig.add_subplot(gs[0, 1]),  # (b) FFT-based ITPC
    fig.add_subplot(gs[1, 0])   # (c) Wavelet-based ITPC
]
fig.add_subplot(gs[1, 1]).axis('off')  # Empty space

fig.suptitle("Test–Retest Reliability of WLI Measures (R1 vs R2, Outliers Removed)", fontsize=15)

# === Loop through methods in desired order ===
for i, method in enumerate(methods):
    ax = axs[i]
    x = pivot_df[(method, 'R1')]
    y = pivot_df[(method, 'R2')]

    # Correlation stats
    r, p = pearsonr(x, y)
    n = len(x)
    dfree = n - 2
    t_val = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2) if r**2 < 1 else np.inf

    # Plot regression with scatter
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.7})
    ax.set_xlabel(f"{labels[method]} (R1)", fontsize=10)
    ax.set_ylabel(f"{labels[method]} (R2)", fontsize=10)
    ax.set_title(f"r({dfree}) = {r:.2f}, t = {t_val:.2f}, p = {p:.3f}, n = {n}", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel label
    ax.text(-0.1, 1.1, panel_labels[i], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# With outliers removed — using FIRST(OTHER) vs SECOND(R2)

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Load the Excel file
file_path = r"filtered_results_withR2.xlsx"
df = pd.read_excel(file_path)

# === Build FIRST (OTHER) vs SECOND (R2) pairs ===
# Keep only subjects that have an R2 session
subjects_with_r2 = df.loc[df['SESSION'] == 'R2', 'SUB'].unique()

# Keep only rows for those subjects and sessions of interest
subset = df[
    df['SUB'].isin(subjects_with_r2) &
    df['SESSION'].isin(['R2', 'R1', 'L2'])
].copy()

# Choose the earliest session per subject (R1 preferred over L2)
priority = {'R1': 0, 'L2': 1}
other_df = (
    subset[subset['SESSION'].isin(['R1', 'L2'])]
      .assign(_prio=lambda d: d['SESSION'].map(priority))
      .sort_values(['SUB', '_prio'])
      .groupby('SUB', as_index=False)
      .first()
      .drop(columns=['_prio'])
)

# Get the R2 rows
r2_df = subset[subset['SESSION'] == 'R2'].copy()

# Merge into wide table: _OTHER = FIRST, _R2 = SECOND
paired = r2_df.merge(
    other_df,
    on='SUB',
    how='left',
    suffixes=('_R2', '_OTHER')
)

# Keep only columns of interest, set MultiIndex columns (method, timepoint)
methods = ['WLI_ITPC', 'WLI_POWER', 'WLI_ITPC_WAVELET']
first_cols  = [f'{m}_OTHER' for m in methods]  # FIRST session
second_cols = [f'{m}_R2'    for m in methods]  # SECOND session

pivot_df = (
    paired[['SUB'] + first_cols + second_cols]
      .set_index('SUB')
      .rename(columns={**{f'{m}_OTHER': (m, 'FIRST') for m in methods},
                       **{f'{m}_R2':    (m, 'SECOND') for m in methods}})
)

# Ensure MultiIndex for columns
pivot_df.columns = pd.MultiIndex.from_tuples(pivot_df.columns)

# Drop participants missing either FIRST or SECOND
pivot_df = pivot_df.dropna()

# === Remove outliers using 3SD rule (within each method across both timepoints) ===
for m in methods:
    x = pivot_df[(m, 'FIRST')]
    y = pivot_df[(m, 'SECOND')]
    combined = pd.concat([x, y])
    mean_val = combined.mean()
    std_val  = combined.std()

    keep_mask = (
        x.between(mean_val - 3*std_val, mean_val + 3*std_val) &
        y.between(mean_val - 3*std_val, mean_val + 3*std_val)
    )
    pivot_df = pivot_df[keep_mask]

# === Correlations after outlier removal ===
print("📈 Correlation between FIRST and SECOND session for each WLI method (outliers removed):")
for m in ['WLI_POWER', 'WLI_ITPC', 'WLI_ITPC_WAVELET']:  # print in your preferred order
    x = pivot_df[(m, 'FIRST')]
    y = pivot_df[(m, 'SECOND')]

    n = len(x)
    if n > 2:
        r, p = pearsonr(x, y)
        dfree = n - 2
        t_val = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2) if r**2 < 1 else np.inf
        print(f"{m}: r({dfree}) = {r:.3f}, t = {t_val:.2f}, p = {p:.4f}, n = {n}")
    else:
        print(f"{m}: insufficient pairs after outlier removal (n = {n})")

# === Plot (FIRST vs SECOND, outliers removed) ===
methods_plot = ['WLI_POWER', 'WLI_ITPC', 'WLI_ITPC_WAVELET']  # (a) Power, (b) FFT ITPC, (c) Wavelet ITPC
labels = {
    'WLI_ITPC': 'FFT-based ITPC',
    'WLI_POWER': 'Evoked Power Spectrum',
    'WLI_ITPC_WAVELET': 'Wavelet-based ITPC'
}
panel_labels = ['(a)', '(b)', '(c)']

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2)
axs = [
    fig.add_subplot(gs[0, 0]),  # (a)
    fig.add_subplot(gs[0, 1]),  # (b)
    fig.add_subplot(gs[1, 0])   # (c)
]
fig.add_subplot(gs[1, 1]).axis('off')

fig.suptitle("Test–Retest Reliability (First vs Second Session, Outliers Removed)", fontsize=15)

for i, m in enumerate(methods_plot):
    ax = axs[i]
    x = pivot_df[(m, 'FIRST')]
    y = pivot_df[(m, 'SECOND')]

    # Guard for very small n
    if len(x) > 2:
        r, p = pearsonr(x, y)
        dfree = len(x) - 2
        t_val = (r * np.sqrt(dfree)) / np.sqrt(1 - r**2) if r**2 < 1 else np.inf
        title = f"r({dfree}) = {r:.2f}, t = {t_val:.2f}, p = {p:.3f}, n = {len(x)}"
    else:
        title = f"n = {len(x)} (insufficient for r)"

    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha': 0.7})
    ax.set_xlabel(f"{labels[m]} (First Session)", fontsize=10)
    ax.set_ylabel(f"{labels[m]} (Second Session)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.text(-0.1, 1.1, panel_labels[i],
            transform=ax.transAxes,
            fontsize=13, fontweight='bold',
            va='top', ha='right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Test Retest SICR
import pandas as pd
from scipy.stats import pearsonr

# === Load and filter the data ===
file_path = "DL_scores.xlsx"
df = pd.read_excel(file_path)

# Filter for R1 and R2 sessions
r1_r2_df = df[df['File name'].str.contains('R1|R2', na=False)].copy()

# Extract subject number (SUB) and session (R1 or R2)
r1_r2_df['SUB'] = r1_r2_df['File name'].str.extract(r'SUB(\d+)_')[0]
r1_r2_df['Session'] = r1_r2_df['File name'].str.extract(r'_(R1|R2)_')[0]

# Pivot the table to pair R1 and R2 per subject
pivot_df = r1_r2_df.pivot(index='SUB', columns='Session', values='Learning score (overall)')
pivot_df = pivot_df.dropna(subset=['R1', 'R2'])

# === Compute Pearson correlation ===
r1_scores = pivot_df['R1']
r2_scores = pivot_df['R2']
correlation, p_value = pearsonr(r1_scores, r2_scores)

# Print results
print("Pearson correlation between R1 and R2 learning scores:")
print(f"r = {correlation:.3f}")
print(f"p = {p_value:.3f}")

# Optional: Save paired data
# pivot_df.to_excel("Paired_R1_R2_LearningScores.xlsx")

import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and filter the data ===
file_path = r"DL_scores.xlsx"
df = pd.read_excel(file_path)

# Filter for R1 and R2 sessions
r1_r2_df = df[df['File name'].str.contains('R1|R2', na=False)].copy()

# Extract subject number and session type
r1_r2_df['SUB'] = r1_r2_df['File name'].str.extract(r'SUB(\d+)_')[0]
r1_r2_df['Session'] = r1_r2_df['File name'].str.extract(r'_(R1|R2)_')[0]

# Pivot to get paired scores
pivot_df = r1_r2_df.pivot(index='SUB', columns='Session', values='Learning score (overall)')
pivot_df = pivot_df.dropna(subset=['R1', 'R2'])

# === Correlation analysis ===
r1_scores = pivot_df['R1']
r2_scores = pivot_df['R2']
correlation, p_value = pearsonr(r1_scores, r2_scores)

# === Visualization ===
plt.figure(figsize=(8, 6))
sns.regplot(x=r1_scores, y=r2_scores, ci=None, scatter_kws={'s': 60})
plt.xlabel("Learning Score (R1)")
plt.ylabel("Learning Score (R2)")
plt.title(f"Correlation between R1 and R2 Scores\nr = {correlation:.2f}, p = {p_value:.3f}")
plt.grid(True)
plt.tight_layout()
plt.show()














