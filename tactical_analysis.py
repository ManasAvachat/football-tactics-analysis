import pandas as pd
import numpy as np

df = pd.read_csv('data/players_data-2025_2026.csv')

# Filtering out players who haven't played enough (minimum 5 full games)
df = df[df['90s'] >= 5].copy() 

# Defining Positives and Negatives using se a mix of goalscoring, playmaking, defending, and goalkeeping
# Adjusted to match your specific column list (e.g., SoT, Fls, TklW)
positive_metrics = ['Gls', 'Ast', 'SoT', 'Int', 'TklW', 'Saves', 'CS']
negative_metrics = ['Fls', 'CrdY', 'CrdR']

# Converting all raw totals into "Per 90" stats for fairness
all_metrics = positive_metrics + negative_metrics
for col in all_metrics:
    df[col + '_per90'] = df[col] / df['90s']

# Standardize (Z-Score) the metrics within each position comparing Forwards to Forwards, and Defenders to Defenders
def calculate_zscore(series):
    if series.std() == 0 or np.isnan(series.std()): 
        return 0
    return (series - series.mean()) / series.std()

for col in all_metrics:
    per90_col = col + '_per90'
    z_col = col + '_z'
    df[z_col] = df.groupby('Pos')[per90_col].transform(calculate_zscore)

# Calculate the Comprehensive Net Impact Score Positives add to the score, Negatives deduct from the score
def calculate_net_impact(row):
    pos = str(row['Pos'])
    
    # Calculate the penalty for mistakes (Applies to everyone) 
    # Fouls and cards are deducted here
    mistake_penalty = (row['Fls_z'] * 0.5) + (row['CrdY_z'] * 1.0) + (row['CrdR_z'] * 3.0)
    
    # Calculate the positive impact based on position
    if 'FW' in pos:
        positives = (row['Gls_z'] * 2.0) + (row['Ast_z'] * 1.0) + (row['SoT_z'] * 0.5)
    elif 'MF' in pos:
        positives = (row['Ast_z'] * 1.5) + (row['Int_z'] * 1.0) + (row['TklW_z'] * 1.0)
    elif 'DF' in pos:
        positives = (row['Int_z'] * 1.5) + (row['TklW_z'] * 1.5) + (row['Gls_z'] * 0.5)
    elif 'GK' in pos:
        positives = (row['Saves_z'] * 2.0) + (row['CS_z'] * 1.5)
    else:
        positives = 0
        
    return round((positives - mistake_penalty), 2)

df['Net_Impact_Score'] = df.apply(calculate_net_impact, axis=1)

print("\n--- Top 10 Game Changers (Net Impact Model) ---")
top_players = df.sort_values(by='Net_Impact_Score', ascending=False)
print(top_players[['Player', 'Pos', 'Squad', 'Net_Impact_Score']].head(10))

# Let's see the ceiling: Average score of the Top 5 players in each position
print("\n--- 'The Elite Ceiling' (Avg Score of Top 5 Players per Position) ---")
# We group by position, grab the top 5 scores, then average them
elite_avg = df.groupby('Pos')['Net_Impact_Score'].nlargest(5).groupby('Pos').mean().round(2)
print(elite_avg.sort_values(ascending=False))

# Let's see the distribution: Which position dominates the Top 50 overall?
print("\n--- Positions of the Top 50 Game Changers ---")
top_50 = df.nlargest(50, 'Net_Impact_Score')
print(top_50['Pos'].value_counts())