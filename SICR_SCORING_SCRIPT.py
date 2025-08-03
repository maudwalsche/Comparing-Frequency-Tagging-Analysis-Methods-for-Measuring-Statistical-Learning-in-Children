# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 21:13:18 2025

@author: User
"""

# This script calculates relative similarity scores for targets, foils, long targets, short targets, long foils, and short foils
# based on the Longest Common Subsequence and the Damerau-Levenshtein distance.

import os, glob
import pandas as pd


file_path = r"C:\Users\User\OneDrive - UGent\TULIP - Towards Understanding Learning In Preterms - Data (No personal identifiers!)\SICR\SICR_excel_data\SICR_excels - clean copy (backup - don't edit)"
output_path = r"C:\Users\User\OneDrive - UGent\Bureaublad\Thesis Data Processing\SICR DATA\SICR SCORED2"

os.chdir(file_path)
files =  glob.glob('TULIP_SUB*.xlsx')

save_output = True

syllable_list = ['tu', 'bi', 'po', 'ka', 'le', 'vi', 'so', 'gu', 'ma', 'no', 'mu', 'be']
target_idx_ses1 = [0, 1, 6, 7, 9, 10, 13, 15, 16, 19, 20, 24, 25, 26, 27, 31]
target_idx_ses2 = [0, 3, 4, 6, 7, 8, 10, 12, 15, 17, 21, 22, 24, 27, 28, 30]

def remove_rep_hes(resp):
    syllables = resp.lower().split()
    result = []
    for i in range(len(syllables)):
        if result and syllables[i] == result[-1]:  # repetitions
            continue
        if i + 1 < len(syllables) and syllables[i] in syllables[i + 1]:  # hesitations
            continue
        result.append(syllables[i])
    
    return ' '.join(result)

def check_and_replace_syl(syl):
    if len(syl) == 3:
        for i in range(3):
            modified_syl = syl[:i] + syl[i+1:]
            if modified_syl in syllable_list:
                return modified_syl
    return syl

def blended_phonemes(resp):
    syllables = resp.lower().split()
    return ' '.join(check_and_replace_syl(syl) for syl in syllables)

def syllable_mapping(sequences, responses):
    mapping = {s: {} for s in syllable_list}
    
    for seq, resp in zip(sequences, responses):
        syl_seq = seq.lower().split()
        syl_resp = remove_rep_hes(blended_phonemes(resp)).split()  # use clean responses
        
        for s in syl_seq:
            for r in syl_resp:
                if set(s) & set(r): 
                    mapping[s][r] = mapping[s].get(r, 0) + 1
    
    # remove responses with count of 1 (= noise) unless for key syllable
    # count of key syllable first and then the others ascending order (for readability)
    for s in mapping:
        items = [(r, c) for r, c in mapping[s].items() if c > 1 or r == s]
        items = sorted(items, key=lambda x: (-1 if x[0] == s else 0, -x[1]))
        mapping[s] = dict(items)
    
    return mapping

def perception_effects(responses, mapping, threshold=0.75):
    perception_eff = {s: s for s in syllable_list}
    
    for syl, resp_counts in mapping.items():
        total = sum(resp_counts.values())        
        for resp, count in resp_counts.items():
            if resp != syl and count / total > threshold:
                print(f"\n'{syl}' likely has a dominant response '{resp}' "
                      f"({count}/{total} = {round(count / total * 100, 2)}%)")
                answer = input(f"Do you want to replace '{resp}' with '{syl}'? (y/n): ")
                
                if answer == 'y':
                    perception_eff[resp] = syl
                    print(f"Replaced '{resp}' with '{syl}'.")
                else:
                    perception_eff[syl] = syl
                    print('Kept original.')
    
    updated_responses = []
    for resp in responses:
        syl_resp = remove_rep_hes(blended_phonemes(resp)).split()  # use clean responses
        updated_responses.append(' '.join([perception_eff.get(r, r) for r in syl_resp ]))
    
    return updated_responses

def LCS(seq1, seq2):
    ''' Longest Common Subsequence '''
    
    seq1 = seq1.lower().split()
    seq2 = seq2.lower().split()
    
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    
    return dp[m][n], dp[m][n] / m

def DL(seq1, seq2):
    ''' Damerau-Levenshtein '''
    
    seq1 = seq1.lower().split()
    seq2 = seq2.lower().split()
    
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
            
            if i > 1 and j > 1 and seq1[i - 1] == seq2[j - 2] and seq1[i - 2] == seq2[j - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)  # transposition
    
    return dp[m][n], 1 - dp[m][n] / max(m, n)


summary_df_LCS = pd.DataFrame(columns=['File name', 'Mean LCS similarity - targets (short)', 
                                      'Mean LCS similarity - foils (short)', 'Learning score (short)', 
                                      'Mean LCS similarity - targets (long)', 'Mean LCS similarity - foils (long)', 
                                      'Learning score (long)', 'Mean LCS similarity - targets (overall)', 
                                      'Mean LCS similarity - foils (overall)', 'Learning score (overall)'])

summary_df_DL = pd.DataFrame(columns=['File name', 'Mean DL similarity - targets (short)', 
                                      'Mean DL similarity - foils (short)', 'Learning score (short)', 
                                      'Mean DL similarity - targets (long)', 'Mean DL similarity - foils (long)', 
                                      'Learning score (long)', 'Mean DL similarity - targets (overall)', 
                                      'Mean DL similarity - foils (overall)', 'Learning score (overall)'])


for file in files:
    print(file)
    
    df = pd.read_excel(file, header=0)
    sequences = df.iloc[:, 0].astype(str).values
    responses = df.iloc[:, 1].fillna('').astype(str).values
    
    scores_LCS = {'target': 0, 'foil': 0, 'short_target': 0, 'short_foil': 0, 'long_target': 0, 'long_foil': 0}
    counts_LCS = {'target': 0, 'foil': 0, 'short_target': 0, 'short_foil': 0, 'long_target': 0, 'long_foil': 0}
    
    scores_DL = {'target': 0, 'foil': 0, 'short_target': 0, 'short_foil': 0, 'long_target': 0, 'long_foil': 0}
    counts_DL = {'target': 0, 'foil': 0, 'short_target': 0, 'short_foil': 0, 'long_target': 0, 'long_foil': 0}
    
    results = []
    
    # repetitions/hesitations and blended consonants happens within
    # syllable_mapping and perception_effects functions
    mapping = syllable_mapping(sequences, responses)
    cleaned_responses = perception_effects(responses, mapping, threshold=0.75)
    
    # threshold of 75% does the job of getting the obvious ones out
    # maybe also try a lower threshold because one-to-one mapping is not perfect
    # other responses may be mapped to a syllable (e.g. 'ka' and 'ma' both get assigned 'na',
    # which is more likely to be a response to 'ma' than 'ka')
    
    for i in range(32):
        LCS_score, LCS_rel_sim = LCS(sequences[i], cleaned_responses[i])
        DL_distance, DL_rel_sim = DL(sequences[i], cleaned_responses[i])
        results.append([cleaned_responses[i], LCS_score, LCS_rel_sim, DL_distance, DL_rel_sim])
        
        if i in target_idx_ses1:
            scores_LCS['target'] += LCS_rel_sim
            counts_LCS['target'] += 1
            if len(sequences[i].split()) == 3:
                scores_LCS['short_target'] += LCS_rel_sim
                counts_LCS['short_target'] += 1
            else:
                scores_LCS['long_target'] += LCS_rel_sim
                counts_LCS['long_target'] += 1
        else:
            scores_LCS['foil'] += LCS_rel_sim
            counts_LCS['foil'] += 1
            if len(sequences[i].split()) == 3:
                scores_LCS['short_foil'] += LCS_rel_sim
                counts_LCS['short_foil'] += 1
            else:
                scores_LCS['long_foil'] += LCS_rel_sim
                counts_LCS['long_foil'] += 1
                
        if i in target_idx_ses1:
            scores_DL['target'] += DL_rel_sim
            counts_DL['target'] += 1
            if len(sequences[i].split()) == 3:
                scores_DL['short_target'] += DL_rel_sim
                counts_DL['short_target'] += 1
            else:
                scores_DL['long_target'] += DL_rel_sim
                counts_DL['long_target'] += 1
        else:
            scores_DL['foil'] += DL_rel_sim
            counts_DL['foil'] += 1
            if len(sequences[i].split()) == 3:
                scores_DL['short_foil'] += DL_rel_sim
                counts_DL['short_foil'] += 1
            else:
                scores_DL['long_foil'] += DL_rel_sim
                counts_DL['long_foil'] += 1
    
    averages_LCS = {key: (scores_LCS[key] / counts_LCS[key] if counts_LCS[key] > 0 else 0) for key in scores_LCS}
    averages_DL = {key: (scores_DL[key] / counts_DL[key] if counts_DL[key] > 0 else 0) for key in scores_DL}
    
    learning_scores_LCS = {
        'overall': averages_LCS['target'] - averages_LCS['foil'],
        'short': averages_LCS['short_target'] - averages_LCS['short_foil'],
        'long': averages_LCS['long_target'] - averages_LCS['long_foil']
    }
    
    learning_scores_DL = {
        'overall': averages_DL['target'] - averages_DL['foil'],
        'short': averages_DL['short_target'] - averages_DL['short_foil'],
        'long': averages_DL['long_target'] - averages_DL['long_foil']
    }
    
    results_df = pd.DataFrame(results, columns=['Cleaned Response', 'LCS Score', 'LCS Relative Similarity', 
                                                'DL Distance', 'DL Relative Similarity'])
    df[['Cleaned Response', 'LCS Score', 'LCS Relative Similarity', 'DL Distance', 'DL Relative Similarity']] = results_df
    if save_output:
        df.to_excel(f'{output_path}/{file[0:-5]}_scored.xlsx', index=False)
    
    new_row_LCS = [file, averages_LCS['short_target'], averages_LCS['short_foil'],
                   learning_scores_LCS['short'], averages_LCS['long_target'],
                   averages_LCS['long_foil'], learning_scores_LCS['long'],
                   averages_LCS['target'], averages_LCS['foil'], learning_scores_LCS['overall']]
    summary_df_LCS.loc[len(summary_df_LCS)] = new_row_LCS
    
    new_row_DL = [file, averages_DL['short_target'], averages_DL['short_foil'],
                  learning_scores_DL['short'], averages_DL['long_target'],
                  averages_DL['long_foil'], learning_scores_DL['long'],
                  averages_DL['target'], averages_DL['foil'], learning_scores_DL['overall']]
    summary_df_DL.loc[len(summary_df_DL)] = new_row_DL

if save_output:
    summary_df_LCS.to_excel(f'{output_path}/LCS_scores.xlsx', index=False)
    summary_df_DL.to_excel(f'{output_path}/DL_scores.xlsx', index=False)

print('Done')
