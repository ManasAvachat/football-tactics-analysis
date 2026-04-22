import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def apply_xV_model(df):
    df = df.copy().fillna(0)
    # Restored Saves and CS so Goalkeepers get proper baseline scores
    metrics = ['Gls', 'Ast', 'KP', 'PrgP', 'Prog', 'SoT', 'Int', 'TklW', 'Blocks', 'Clr', 'Saves', 'CS', 'Fls', 'CrdY', 'CrdR', 'OG']
    valid_metrics = [c for c in metrics if c in df.columns]
    
    for col in valid_metrics:
        df[col + '_per90'] = df[col] / df['90s']
        
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

print("Loading and Formatting Historical Data...")
df_24 = apply_xV_model(pd.read_csv('data/players_data-2024_2025.csv')[lambda x: x['90s'] >= 10])
df_25 = apply_xV_model(pd.read_csv('data/players_data-2025_2026.csv')[lambda x: x['90s'] >= 10])

# --- 1. TRAIN OUTFIELD MODEL ---
merged_out = pd.merge(df_24[df_24['Pos'] != 'GK'], df_25[df_25['Pos'] != 'GK'][['Player', 'Expected_Value_Added']], on='Player', suffixes=('', '_NextYear'))
prog_feat = 'PrgP_per90' if 'PrgP_per90' in merged_out.columns else 'Prog_per90'
features_out = ['Age', '90s', 'Gls_per90', prog_feat, 'TklW_per90', 'Expected_Value_Added']

X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(merged_out[features_out], merged_out['Expected_Value_Added_NextYear'], test_size=0.2, random_state=42)
model_out = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train_out, y_train_out)
joblib.dump(model_out, 'wc_2026_model_outfield.pkl')
print(f"Outfield AI saved! Testing Error (MSE): {mean_squared_error(y_test_out, model_out.predict(X_test_out)):.3f}")

# --- 2. TRAIN GOALKEEPER MODEL ---
merged_gk = pd.merge(df_24[df_24['Pos'] == 'GK'], df_25[df_25['Pos'] == 'GK'][['Player', 'Expected_Value_Added']], on='Player', suffixes=('', '_NextYear'))
features_gk = ['Age', '90s', 'Saves_per90', 'CS_per90', 'Expected_Value_Added']

# GKs have a smaller dataset, so we train on what we have
X_train_gk, X_test_gk, y_train_gk, y_test_gk = train_test_split(merged_gk[features_gk], merged_gk['Expected_Value_Added_NextYear'], test_size=0.2, random_state=42)
model_gk = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train_gk, y_train_gk)
joblib.dump(model_gk, 'wc_2026_model_gk.pkl')
print(f"Goalkeeper AI saved! Testing Error (MSE): {mean_squared_error(y_test_gk, model_gk.predict(X_test_gk)):.3f}")