import pandas as pd
import numpy as np

# 1. Load Data and Filter for established starters
df = pd.read_csv('data/players_data-2025_2026.csv')
df = df[df['90s'] >= 10].copy() 

# 2. Separate Volume Metrics (need /90 division) from Rate Metrics (already averages)
volume_metrics = ['Gls', 'Ast', 'KP', 'PrgP', 'SoT', 'Int', 'TklW', 'Blocks', 'Clr', 'Saves', 'CS', 'Fls', 'CrdY', 'CrdR', 'Off', 'OG']
rate_metrics = ['PPM', '+/-90'] # Points Per Match & Plus/Minus per 90

v_mets = [c for c in volume_metrics if c in df.columns]
r_mets = [c for c in rate_metrics if c in df.columns]

# 3. Convert Volume stats to Per 90
for col in v_mets:
    df[col + '_per90'] = df[col] / df['90s']

# 4. Standardize (Z-Score)
def calculate_zscore(series):
    if series.std() == 0 or np.isnan(series.std()): 
        return 0
    return (series - series.mean()) / series.std()

# Apply Z-score to Per 90 volume columns
for col in v_mets:
    df[col + '_z'] = df.groupby('Pos')[col + '_per90'].transform(calculate_zscore)

# Apply Z-score directly to Rate columns
for col in r_mets:
    df[col + '_z'] = df.groupby('Pos')[col].transform(calculate_zscore)

def get_val(row, col_name):
    return row.get(col_name + '_z', 0)

# 5. The Master Formula (Every position's positive weights sum to ~4.5)
def calculate_net_impact(row):
    pos = str(row['Pos'])
    
    # Mistakes: Max penalty pool reduced slightly to avoid breaking the model
    mistake_penalty = (get_val(row, 'Fls') * 0.3) + (get_val(row, 'CrdY') * 0.5) + \
                      (get_val(row, 'CrdR') * 2.0) + (get_val(row, 'Off') * 0.3) + (get_val(row, 'OG') * 2.0)
    
    # Forwards (Total Weight: 4.5) -> High reward for shooting, scoring, and actual winning (+/-90)
    if 'FW' in pos:
        positives = (get_val(row, 'Gls') * 1.5) + (get_val(row, 'SoT') * 1.0) + \
                    (get_val(row, 'Ast') * 1.0) + (get_val(row, '+/-90') * 1.0)
                    
    # Midfielders (Total Weight: 4.5) -> High reward for progression, chance creation, and winning
    elif 'MF' in pos:
        positives = (get_val(row, 'PrgP') * 1.2) + (get_val(row, 'KP') * 1.2) + \
                    (get_val(row, 'Ast') * 0.8) + (get_val(row, 'TklW') * 0.5) + \
                    (get_val(row, '+/-90') * 0.8)
                    
    # Defenders (Total Weight: 4.5) -> Massive nerf to raw defending volume. 
    # High reward for Plus/Minus (clean sheets/winning) and Progression (build-up play).
    elif 'DF' in pos:
        positives = (get_val(row, '+/-90') * 1.5) + (get_val(row, 'PrgP') * 1.2) + \
                    (get_val(row, 'Int') * 0.8) + (get_val(row, 'TklW') * 0.8) + \
                    (get_val(row, 'PPM') * 0.2)
                    
    # Goalkeepers (Total Weight: 4.5)
    elif 'GK' in pos:
        positives = (get_val(row, 'Saves') * 1.5) + (get_val(row, 'CS') * 1.5) + \
                    (get_val(row, '+/-90') * 1.5)
    else:
        positives = 0
        
    return round((positives - mistake_penalty), 2)

df['Net_Impact_Score'] = df.apply(calculate_net_impact, axis=1)

# 6. Clean Output
print("\n--- The True Game Changers (Anchored to Team Success) ---")
valid_display = [c for c in ['Player', 'Pos', 'Squad', 'Net_Impact_Score'] if c in df.columns]
print(df.sort_values(by='Net_Impact_Score', ascending=False)[valid_display].head(10).to_string(index=False))

print("\n--- 'The Elite Ceiling' (Avg Score of Top 5 Players per Position) ---")
elite_avg = df.groupby('Pos')['Net_Impact_Score'].nlargest(5).groupby('Pos').mean().round(2)
# Added the sort_values command before converting to string
print(elite_avg.sort_values(ascending=False).to_string())