# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 21:05:43 2025

@author: User
"""
import pandas as pd

# === Define file paths ===
file1_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\SICR DATA\SICR SCORED\DL_scores.xlsx"
file2_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\SICR DATA\SICR SCORED2\DL_scores.xlsx"

# === Load both files ===
df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

# === Rename the score column for clarity ===
score_col = 'Mean DL similarity - targets (overall)'

# === Keep only relevant columns ===
df1 = df1[['File name', score_col]].rename(columns={score_col: 'Score_file1'})
df2 = df2[['File name', score_col]].rename(columns={score_col: 'Score_file2'})

# === Merge on "File name" to keep only matching rows ===
merged = pd.merge(df1, df2, on='File name', how='inner')

# === Result: merged contains only matching filenames with both scores ===
print(merged.head())

# === Optional: Save to CSV or Excel if needed ===
# merged.to_csv("matched_scores.csv", index=False)

import numpy as np

# === Drop any missing values just in case ===
merged = merged.dropna(subset=['Score_file1', 'Score_file2'])

# === Convert to NumPy array for ICC calculation ===
scores = merged[['Score_file1', 'Score_file2']].to_numpy()
n_subjects, n_raters = scores.shape

# === Compute ICC(2,1) manually ===
subject_means = scores.mean(axis=1, keepdims=True)
grand_mean = scores.mean()

SS_total = ((scores - grand_mean) ** 2).sum()
SS_subject = n_raters * ((subject_means - grand_mean) ** 2).sum()
SS_rater = n_subjects * ((scores.mean(axis=0) - grand_mean) ** 2).sum()
SS_error = SS_total - SS_subject - SS_rater

df_subject = n_subjects - 1
df_rater = n_raters - 1
df_error = df_subject * df_rater

MS_subject = SS_subject / df_subject
MS_error = SS_error / df_error

icc_2_1 = (MS_subject - MS_error) / (MS_subject + (n_raters - 1) * MS_error)

# === Print the ICC result ===
print(f"ICC(2,1) = {icc_2_1:.4f}")








