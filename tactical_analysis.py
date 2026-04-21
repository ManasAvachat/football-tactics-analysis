import pandas as pd
import numpy as np

# 1. Load and Filter
df = pd.read_csv('data/players_data-2025_2026.csv')
df = df[df['90s'] >= 5].copy() 

# 2. Expanded Metrics for Deeper Realism
positive_metrics = ['Gls', 'Ast', 'KP', 'PrgP', 'SoT', 'Int', 'TklW', 'Blocks', 'Clr', 'Saves', 'CS']
negative_metrics = ['Fls', 'CrdY', 'CrdR', 'Off', 'OG']

all_metrics = [c for c in positive_metrics + negative_metrics if c in df.columns]

for col in all_metrics:
    df[col + '_per90'] = df[col] / df['90s']

def calculate_zscore(series):
    if series.std() == 0 or np.isnan(series.std()): 
        return 0
    return (series - series.mean()) / series.std()

for col in all_metrics:
    z_col = col + '_z'
    df[z_col] = df.groupby('Pos')[col + '_per90'].transform(calculate_zscore)

# Helper function to safely get z-scores
def get_val(row, col_name):
    return row.get(col_name + '_z', 0)

# Helper function to safely get z-scores
def get_val(row, col_name):
    return row.get(col_name + '_z', 0)

# 5. The Re-Balanced Impact Formula
def calculate_net_impact(row):
    pos = str(row['Pos'])
    
    # Enhanced Penalties: Own goals are a disaster, Offsides kill momentum
    mistake_penalty = (get_val(row, 'Fls') * 0.5) + (get_val(row, 'CrdY') * 1.0) + \
                      (get_val(row, 'CrdR') * 3.0) + (get_val(row, 'Off') * 0.5) + (get_val(row, 'OG') * 3.0)
    
    # Rebalanced Positives: Midfielders and Defenders get rewarded for their specific duties
    if 'FW' in pos:
        positives = (get_val(row, 'Gls') * 1.5) + (get_val(row, 'Ast') * 1.0) + (get_val(row, 'SoT') * 0.5)
    elif 'MF' in pos:
        # Midfielders are the engine: High reward for moving the ball forward and creating chances
        positives = (get_val(row, 'PrgP') * 1.5) + (get_val(row, 'KP') * 1.5) + \
                    (get_val(row, 'Ast') * 1.0) + (get_val(row, 'TklW') * 0.5)
    elif 'DF' in pos:
        # Defenders now have 4 ways to score points instead of 2
        positives = (get_val(row, 'Int') * 1.5) + (get_val(row, 'TklW') * 1.5) + \
                    (get_val(row, 'Blocks') * 1.0) + (get_val(row, 'Clr') * 1.0)
    elif 'GK' in pos:
        positives = (get_val(row, 'Saves') * 2.0) + (get_val(row, 'CS') * 1.5)
    else:
        positives = 0
        
    return round((positives - mistake_penalty), 2)

df['Net_Impact_Score'] = df.apply(calculate_net_impact, axis=1)

# 6. Clean Output Generation
print("\n--- Top 10 Game Changers (Re-Balanced Model) ---")
valid_display = [c for c in ['Player', 'Pos', 'Squad', 'Net_Impact_Score'] if c in df.columns]
# .to_string(index=False) hides the random row numbers for a cleaner look
print(df.sort_values(by='Net_Impact_Score', ascending=False)[valid_display].head(10).to_string(index=False))

print("\n--- 'The Elite Ceiling' (Avg Score of Top 5 Players per Position) ---")
elite_avg = df.groupby('Pos')['Net_Impact_Score'].nlargest(5).groupby('Pos').mean().round(2)
# .to_string() removes the annoying "Name: Net_Impact_Score, dtype: float64" bug
print(elite_avg.to_string())