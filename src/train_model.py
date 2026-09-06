"""
WR Breakout Model -- Training

Builds features, constructs the breakout label, and trains the model, all
in one pass. Run this once (or whenever you want to retrain on the latest
historical data) before running predict_2026.py.

Methodology summary:
- Position: WR only.
- Label: a jump of at least 25 percentile points in PPR points-per-game
  rank within the WR position and season, landing in the top 30% of WR
  that season. Percentile-based (not a fixed rank-improvement threshold)
  specifically to avoid biasing the label toward larger position pools.
- Features: target share, air yards share, WOPR, RACR, snap %, age, years
  of experience, draft capital, PPR points per game, PPR points per target
  (per-opportunity efficiency, independent of overall volume), team-relative
  role (target-share rank within team -- real "WR1 vs WR2" status, not
  nflverse's unusable `depth_chart_position` field, which just repeats the
  position name), and NFL Combine athletic testing (40-yard dash, vertical,
  broad jump, cone, shuttle) as a skill signal independent of both usage
  and draft capital.

Output: ../models/wr_breakout_model.joblib
"""
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

YEARS = list(range(2013, 2025))

# ============================================================
# STEP 1: Pull data and build season-level features
# ============================================================
print("Pulling weekly stats, rosters, snap counts, combine data...")
weekly = nfl.import_weekly_data(YEARS)
rosters = nfl.import_seasonal_rosters(YEARS)
snaps = nfl.import_snap_counts(YEARS)
combine = nfl.import_combine_data(list(range(2000, 2025)))

weekly = weekly[weekly['season_type'] == 'REG']
# Team-target denominator must include ALL pass-catchers (WR+RB+TE), not just WR --
# filtering to WR before this point silently shrinks the denominator and inflates every
# player's target_share. Filter down to WR only AFTER team totals are computed.
weekly_all_pos = weekly[weekly['position'].isin(['WR', 'RB', 'TE'])].copy()

# Rate stats (target_share, air_yards_share, etc.) must be computed from SEASON-LEVEL
# TOTALS, not by averaging each week's ratio -- a player with mostly zero-target games
# and one fluke target gets a nonsense season "rate" otherwise.
team_week = (
    weekly_all_pos.groupby(['recent_team', 'season', 'week'])
    .agg(team_targets=('targets', 'sum'), team_air_yards=('receiving_air_yards', 'sum'))
    .reset_index()
)
weekly_wr = weekly[weekly['position'] == 'WR'].merge(team_week, on=['recent_team', 'season', 'week'], how='left')

agg_funcs = {
    'targets': 'sum', 'receptions': 'sum', 'receiving_yards': 'sum', 'receiving_tds': 'sum',
    'receiving_air_yards': 'sum', 'receiving_yards_after_catch': 'sum',
    'carries': 'sum', 'rushing_yards': 'sum', 'rushing_tds': 'sum',
    'fantasy_points_ppr': 'sum', 'team_targets': 'sum', 'team_air_yards': 'sum', 'week': 'nunique',
}
season = (
    weekly_wr.groupby(['player_id', 'player_display_name', 'position', 'season', 'recent_team'])
    .agg(agg_funcs).reset_index()
    .rename(columns={'week': 'games_played', 'recent_team': 'team'})
)
season['target_share'] = season['targets'] / season['team_targets'].replace(0, np.nan)
season['air_yards_share'] = season['receiving_air_yards'] / season['team_air_yards'].replace(0, np.nan)
season['racr'] = (season['receiving_yards'] / season['receiving_air_yards'].replace(0, np.nan)).clip(-2, 5)
season['wopr'] = 1.5 * season['target_share'] + 0.7 * season['air_yards_share']
season['ppr_ppg'] = season['fantasy_points_ppr'] / season['games_played'].replace(0, np.nan)
season['ppr_per_target'] = season['fantasy_points_ppr'] / season['targets'].replace(0, np.nan)
season = season[season['games_played'] >= 6]

# Snap share
snaps_off = snaps.groupby(['pfr_player_id', 'season'])['offense_pct'].mean().reset_index()
snaps_off = snaps_off.rename(columns={'offense_pct': 'avg_snap_pct'})
id_bridge = rosters[['player_id', 'pfr_id', 'season']].drop_duplicates()
snaps_off = snaps_off.merge(id_bridge, left_on=['pfr_player_id', 'season'], right_on=['pfr_id', 'season'], how='left')
season = season.merge(snaps_off[['player_id', 'season', 'avg_snap_pct']], on=['player_id', 'season'], how='left')

# Age, experience, draft capital
ros_cols = rosters[['player_id', 'season', 'age', 'years_exp', 'draft_number']].drop_duplicates(subset=['player_id', 'season'])
season = season.merge(ros_cols, on=['player_id', 'season'], how='left')
season['draft_number'] = season['draft_number'].fillna(300)

# Team-relative role (real usage-based pecking order, not nflverse's unusable
# depth_chart_position field)
season['team_target_rank'] = season.groupby(['team', 'season'])['target_share'].rank(ascending=False, method='min')
season['team_pos_rank'] = season.groupby(['team', 'season', 'position'])['target_share'].rank(ascending=False, method='min')

# NFL Combine skill data. IMPORTANT: filter out blank pfr_id on BOTH sides before merging --
# a naive join here once blew up a clean 8,320-row combine dataset into 9.9 million rows
# because blank ID strings were matching each other as if they were a real shared key.
rosters_clean = rosters[rosters['pfr_id'].notna() & (rosters['pfr_id'] != '')]
combine_clean = combine[combine['pfr_id'].notna() & (combine['pfr_id'] != '')]
id_bridge_combine = rosters_clean[['player_id', 'pfr_id']].drop_duplicates(subset=['player_id'])
combine_bridged = combine_clean.merge(id_bridge_combine, on='pfr_id', how='inner')
combine_feats = combine_bridged[['player_id', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'wt']].drop_duplicates(subset=['player_id']).rename(columns={'wt': 'combine_weight'})
season = season.merge(combine_feats, on='player_id', how='left')

print(f"Season-level table: {season.shape}, combine coverage: {season['forty'].notna().mean():.1%}")

# ============================================================
# STEP 2: Build the Year N -> Year N+1 training table with the breakout label
# ============================================================
season['pct_rank'] = season.groupby('season')['ppr_ppg'].rank(pct=True)
season['pos_rank'] = season.groupby('season')['ppr_ppg'].rank(ascending=False, method='min')

FEATURE_COLS = [
    'target_share', 'air_yards_share', 'wopr', 'racr', 'avg_snap_pct', 'age', 'years_exp',
    'draft_number', 'games_played', 'ppr_ppg', 'pos_rank', 'receiving_air_yards',
    'receiving_yards_after_catch', 'carries', 'rushing_yards', 'team_target_rank',
    'team_pos_rank', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'combine_weight',
    'ppr_per_target',
]

rows = []
for pid, grp in season.sort_values('season').groupby('player_id'):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - 1):
        yr_n, yr_n1 = grp.loc[i], grp.loc[i + 1]
        if yr_n1['season'] - yr_n['season'] != 1:
            continue
        pct_jump = yr_n1['pct_rank'] - yr_n['pct_rank']
        breakout = int((pct_jump >= 0.25) and (yr_n1['pct_rank'] >= 0.70))
        row = {'player_id': pid, 'season_n': yr_n['season'], 'breakout': breakout}
        for c in FEATURE_COLS:
            row[c] = yr_n[c]
        rows.append(row)

train_df = pd.DataFrame(rows)
train_df = train_df.dropna(subset=['target_share', 'air_yards_share', 'wopr', 'racr'])
print(f"Training table: {train_df.shape}, breakout rate: {train_df['breakout'].mean():.3f}")

# ============================================================
# STEP 3: Train the final model on all available history
# ============================================================
X = train_df[FEATURE_COLS].copy()
y = train_df['breakout']
for col in FEATURE_COLS:
    X[col] = X[col].fillna(X[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_scaled, y)

joblib.dump(
    {'model': model, 'scaler': scaler, 'feature_cols': FEATURE_COLS, 'medians': X.median().to_dict(),
     'combine_raw': combine},  # bundled so predict_2026.py doesn't need to re-pull it
    '../models/wr_breakout_model.joblib',
)
print("\nSaved ../models/wr_breakout_model.joblib")
