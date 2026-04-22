import pandas as pd
import joblib

def apply_xV_model(df):
    df = df.copy().fillna(0)
    metrics = ['Gls', 'Ast', 'KP', 'PrgP', 'Prog', 'SoT', 'Int', 'TklW', 'Blocks', 'Clr', 'Saves', 'CS', 'Fls', 'CrdY', 'CrdR', 'OG']
    valid_metrics = [c for c in metrics if c in df.columns]
    for col in valid_metrics: df[col + '_per90'] = df[col] / df['90s']
    def get_p90(row, col_name): return row.get(col_name + '_per90', 0)
    def calculate_xV(row):
        prog_val = get_p90(row, 'PrgP') if 'PrgP' in df.columns else get_p90(row, 'Prog')
        attack = (get_p90(row, 'Gls') * 1.0) + (get_p90(row, 'Ast') * 1.0) + (get_p90(row, 'SoT') * 0.1)
        buildup = (prog_val * 0.02) + (get_p90(row, 'KP') * 0.1)
        defense = (get_p90(row, 'TklW') * 0.05) + (get_p90(row, 'Int') * 0.05) + (get_p90(row, 'Blocks') * 0.02) + (get_p90(row, 'Clr') * 0.01)
        goalkeeping = (get_p90(row, 'Saves') * 0.25) + (get_p90(row, 'CS') * 0.3)
        mistakes = (get_p90(row, 'OG') * 1.0) + (get_p90(row, 'CrdR') * 1.0) + (get_p90(row, 'CrdY') * 0.05) + (get_p90(row, 'Fls') * 0.02)
        return round((attack + buildup + defense + goalkeeping - mistakes), 3)
    df['Expected_Value_Added'] = df.apply(calculate_xV, axis=1)
    return df

print("Loading Current Data and Applying Filters...")
df_25 = pd.read_csv('data/players_data-2025_2026.csv')
df_25 = df_25[df_25['90s'] >= 10]
df_wc = apply_xV_model(df_25)

wc_nations = [
    'CAN', 'MEX', 'USA', 'AUS', 'IRQ', 'IRN', 'JPN', 'JOR', 'KOR', 'QAT', 'KSA', 'UZB', 
    'ALG', 'CPV', 'COD', 'CIV', 'EGY', 'GHA', 'MAR', 'SEN', 'RSA', 'TUN', 'CUW', 'HAI', 
    'PAN', 'ARG', 'BRA', 'COL', 'ECU', 'PAR', 'URU', 'NZL', 'AUT', 'BEL', 'BIH', 'CRO', 
    'CZE', 'ENG', 'FRA', 'GER', 'NED', 'NOR', 'POR', 'SCO', 'ESP', 'SWE', 'SUI', 'TUR'
]
df_wc['Nation_Code'] = df_wc['Nation'].astype(str).str.split().str[-1]
df_wc = df_wc[df_wc['Nation_Code'].isin(wc_nations)].copy()

# --- PREDICT USING MULTIPLE MODELS ---
model_out = joblib.load('wc_2026_model_outfield.pkl')
model_gk = joblib.load('wc_2026_model_gk.pkl')

df_out = df_wc[df_wc['Pos'] != 'GK'].copy()
df_gk = df_wc[df_wc['Pos'] == 'GK'].copy()

# Outfield
feat_out = model_out.feature_names_in_
for f in feat_out:
    if f not in df_out.columns: df_out[f] = 0.0
df_out['Predicted_2026_xV'] = model_out.predict(df_out[feat_out]).round(3)

# Goalkeepers
feat_gk = model_gk.feature_names_in_
for f in feat_gk:
    if f not in df_gk.columns: df_gk[f] = 0.0
df_gk['Predicted_2026_xV'] = model_gk.predict(df_gk[feat_gk]).round(3)

# Stitch it back together
df_wc = pd.concat([df_out, df_gk])
df_wc['Growth_Delta'] = (df_wc['Predicted_2026_xV'] - df_wc['Expected_Value_Added']).round(3)

# --- PRESENTATION ---
display_cols = ['Player', 'Nation_Code', 'Age', 'Pos', 'Expected_Value_Added', 'Predicted_2026_xV']

print("\n--- Top 15 Overall Predicted 2026 World Cup Game Changers ---")
breakouts = df_wc.sort_values(by='Predicted_2026_xV', ascending=False)
print(breakouts[display_cols].head(15).to_string(index=False))

def get_pos_rank(pos):
    pos_str = str(pos)
    if pos_str == 'FW': return 1
    elif 'FW' in pos_str and 'MF' in pos_str: return 2
    elif pos_str == 'MF': return 3
    elif 'MF' in pos_str and 'DF' in pos_str: return 4
    elif pos_str == 'DF': return 5
    elif pos_str == 'GK': return 6
    return 7

breakouts['Pos_Rank'] = breakouts['Pos'].apply(get_pos_rank)

print("\n--- Top 3 Predicted Game Changers By Position ---")
top_by_pos = breakouts.groupby('Pos').head(3).sort_values(by=['Pos_Rank', 'Predicted_2026_xV'], ascending=[True, False])
print(top_by_pos[display_cols].to_string(index=False))

breakouts.drop(columns=['Pos_Rank']).to_csv('powerbi_wc2026_predictions.csv', index=False)
print("\nExported 'powerbi_wc2026_predictions.csv' for PowerBI Dashboard!")