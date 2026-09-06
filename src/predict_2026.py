"""
WR Breakout Model -- 2026 Prediction

Run this after train_model.py to generate the current shortlist. Pulls live
data at run time, so results reflect whatever nflverse has published as of
right now -- re-running this script on a different day can produce
different results if the underlying season stats get revised or corrected
in the meantime (a real characteristic of depending on live external data,
not a bug).

Two things happen here that are NOT part of the trained model itself:

1. Building 2025 season features (same construction as training, applied
   to the most recently completed season).

2. The current-roster competition check. The trained model only ever sees
   last season's stats, which means it has no way to know a player left a
   team (trade, release, retirement) or that a new competitor arrived (a
   trade-in, a free-agent signing, a draft pick). Confirmed directly: the
   model showed Elic Ayomanor as Tennessee's clear WR1 using 2025 data
   alone, but the real 2026 roster shows Wan'Dale Robinson (an established
   producer, acquired via trade/free agency) and Carnell Tate (the #4
   overall pick in the 2026 draft) had both joined the team -- neither
   visible from 2025 data at all.

   Fix: pull the real, current-season roster, and for each active WR use
   their actual 2025 production (if they played meaningful snaps anywhere)
   or their draft capital (if they're a true rookie) as a competition-
   quality score. A player only clears the shortlist if
   current_team_pos_rank <= 2 -- a genuine path to being at least the
   team's clear #2 option, using real current rosters rather than a
   replayed 2025 depth chart. This is a deployment-time heuristic layered
   on top of the trained model, not something the model itself learned.

Output: ../predictions/wr_2026_shortlist.csv
"""
import pandas as pd
import numpy as np
import joblib

CURRENT_SEASON = 2025     # most recently completed season -- the model's inputs
PREDICTION_SEASON = 2026  # the season being predicted

artifact = joblib.load('../models/wr_breakout_model.joblib')
model, scaler, feature_cols, medians = artifact['model'], artifact['scaler'], artifact['feature_cols'], artifact['medians']
combine_all = artifact['combine_raw']

# ============================================================
# STEP A: Build current-season features (same logic as training)
# ============================================================
weekly = pd.read_parquet(
    f'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{CURRENT_SEASON}.parquet'
)
rosters_now = pd.read_parquet(
    f'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{CURRENT_SEASON}.parquet'
)
snaps_now = pd.read_parquet(
    f'https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{CURRENT_SEASON}.parquet'
)

weekly = weekly[weekly['season_type'] == 'REG']
weekly_all_pos = weekly[weekly['position'].isin(['WR', 'RB', 'TE'])].copy()

team_week = weekly_all_pos.groupby(['team', 'season', 'week']).agg(
    team_targets=('targets', 'sum'), team_air_yards=('receiving_air_yards', 'sum')
).reset_index()
weekly_wr = weekly[weekly['position'] == 'WR'].merge(team_week, on=['team', 'season', 'week'], how='left')

agg_funcs = {
    'targets': 'sum', 'receptions': 'sum', 'receiving_yards': 'sum', 'receiving_tds': 'sum',
    'receiving_air_yards': 'sum', 'receiving_yards_after_catch': 'sum', 'carries': 'sum',
    'rushing_yards': 'sum', 'rushing_tds': 'sum', 'fantasy_points_ppr': 'sum',
    'team_targets': 'sum', 'team_air_yards': 'sum', 'week': 'nunique',
}
season = weekly_wr.groupby(['player_id', 'player_display_name', 'position', 'season', 'team']).agg(
    agg_funcs).reset_index().rename(columns={'week': 'games_played'})
season['target_share'] = season['targets'] / season['team_targets'].replace(0, np.nan)
season['air_yards_share'] = season['receiving_air_yards'] / season['team_air_yards'].replace(0, np.nan)
season['racr'] = (season['receiving_yards'] / season['receiving_air_yards'].replace(0, np.nan)).clip(-2, 5)
season['wopr'] = 1.5 * season['target_share'] + 0.7 * season['air_yards_share']
season['ppr_ppg'] = season['fantasy_points_ppr'] / season['games_played'].replace(0, np.nan)
season['ppr_per_target'] = season['fantasy_points_ppr'] / season['targets'].replace(0, np.nan)
season = season[season['games_played'] >= 6]
season['pos_rank'] = season.groupby('position')['ppr_ppg'].rank(ascending=False, method='min')
season['team_target_rank'] = season.groupby(['team', 'season'])['target_share'].rank(ascending=False, method='min')
season['team_pos_rank'] = season.groupby(['team', 'season', 'position'])['target_share'].rank(ascending=False, method='min')

snaps_off = snaps_now.groupby('pfr_player_id')['offense_pct'].mean().reset_index().rename(columns={'offense_pct': 'avg_snap_pct'})
id_bridge = rosters_now[['gsis_id', 'pfr_id']].drop_duplicates().rename(columns={'gsis_id': 'player_id'})
snaps_off = snaps_off.merge(id_bridge, left_on='pfr_player_id', right_on='pfr_id', how='left')
season = season.merge(snaps_off[['player_id', 'avg_snap_pct']], on='player_id', how='left')

rosters_now['birth_date'] = pd.to_datetime(rosters_now['birth_date'], errors='coerce')
kickoff = pd.Timestamp(f'{PREDICTION_SEASON}-09-10')
rosters_now['age'] = (kickoff - rosters_now['birth_date']).dt.days / 365.25
ros_cols = rosters_now[['gsis_id', 'age', 'years_exp', 'draft_number', 'pfr_id']].drop_duplicates(
    subset=['gsis_id']).rename(columns={'gsis_id': 'player_id'})
season = season.merge(ros_cols, on='player_id', how='left')
season['draft_number'] = season['draft_number'].fillna(300)

combine_clean = combine_all[combine_all['pfr_id'].notna() & (combine_all['pfr_id'] != '')]
combine_feats = combine_clean[['pfr_id', 'forty', 'vertical', 'broad_jump', 'cone', 'shuttle', 'wt']].drop_duplicates(
    subset=['pfr_id']).rename(columns={'wt': 'combine_weight'})
season = season.merge(combine_feats, on='pfr_id', how='left')

# ============================================================
# STEP B: Apply the trained model
# ============================================================
X_pred = season[feature_cols].copy()
for col in feature_cols:
    X_pred[col] = X_pred[col].fillna(medians[col])
season['breakout_probability'] = model.predict_proba(scaler.transform(X_pred))[:, 1]

# ============================================================
# STEP C: Current-roster competition check (see module docstring)
# ============================================================
# Competition now spans ALL pass-catchers (WR+RB+TE), not just other WRs -- a WR's real
# path to targets is blocked just as much by a team's leading pass-catching RB or TE as
# by another WR. Also: players with multiple 2025 team stints (e.g. a mid-season trade)
# must have their stats SUMMED across stints before computing target_share -- an earlier
# version picked up whichever single stint happened to survive a merge, which inflated
# John Metchie III's competition quality using only his highest-share stint (NYJ, 44
# targets) rather than his true, much quieter full season (48 total targets across two
# teams).
weekly_allpos_2025 = weekly_all_pos.copy()

# Merge team_week onto ALL positions (not just WR), then sum both targets and
# team_targets across all of a player's stints (handles mid-season trades) before dividing.
weekly_allpos_2025 = weekly_allpos_2025.merge(team_week, on=['team', 'season', 'week'], how='left')
player_season_allpos = weekly_allpos_2025.groupby(['player_id', 'player_display_name', 'position']).agg(
    targets=('targets', 'sum'), team_targets=('team_targets', 'sum'),
    receiving_yards=('receiving_yards', 'sum'), receiving_tds=('receiving_tds', 'sum'),
    games_played=('week', 'nunique'),
).reset_index()
player_season_allpos['target_share'] = player_season_allpos['targets'] / player_season_allpos['team_targets'].replace(0, np.nan)
# Volume alone (target share) misses efficiency and scoring impact -- a player who's highly
# productive on modest volume, or a big scoring threat, should count as real competition too.
player_season_allpos['rec_yards_per_game'] = player_season_allpos['receiving_yards'] / player_season_allpos['games_played'].replace(0, np.nan)
player_season_allpos['rec_tds_per_game'] = player_season_allpos['receiving_tds'] / player_season_allpos['games_played'].replace(0, np.nan)

# Percentile-normalize each metric across the full league pool (not just within a team)
# before combining -- target_share (0-1), yards/game (0-100+), and TDs/game (0-1ish) are on
# totally different scales, so averaging raw values would let yards/game dominate. Percentile
# rank puts all three on the same 0-1 footing.
player_season_allpos['pct_target_share'] = player_season_allpos['target_share'].rank(pct=True)
player_season_allpos['pct_yards_pg'] = player_season_allpos['rec_yards_per_game'].rank(pct=True)
player_season_allpos['pct_tds_pg'] = player_season_allpos['rec_tds_per_game'].rank(pct=True)
player_season_allpos['composite_quality'] = player_season_allpos[
    ['pct_target_share', 'pct_yards_pg', 'pct_tds_pg']].mean(axis=1)

r_next = pd.read_parquet(
    f'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{PREDICTION_SEASON}.parquet'
)
allpos_active = r_next[(r_next['position'].isin(['WR', 'RB', 'TE'])) & (r_next['status'] == 'ACT')][
    ['gsis_id', 'full_name', 'team', 'position', 'draft_number', 'entry_year']].rename(columns={'gsis_id': 'player_id'})
allpos_active = allpos_active.merge(
    player_season_allpos[['player_id', 'target_share', 'rec_yards_per_game', 'rec_tds_per_game', 'composite_quality']],
    on='player_id', how='left')
allpos_active = allpos_active.drop_duplicates(subset=['player_id'], keep='first')  # one row per player, no stint fan-out

allpos_active['is_rookie'] = allpos_active['entry_year'] == PREDICTION_SEASON
allpos_active['competition_quality'] = allpos_active['composite_quality'].fillna(0)

# Rookie quality, tiered by draft capital -- since rookies have no NFL production yet:
#   - Picks 1-10: 0.95 (pick 1) down to 0.85 (pick 10) -- a truly elite, top-10 pick is
#     presumed capable of clearing even strong incumbent production immediately.
#   - Picks 11-32: 0.80 (pick 11) down to 0.55 (pick 32) -- a real, above-average rookie
#     prior, but not assumed to unseat a genuinely productive incumbent.
#   - Picks 33+: original modest linear formula, floored at 0.02.
# This is a judgment call, not something backtested against real rookie-year outcomes --
# treat it as a reasonable prior, not a validated number.
def rookie_quality(draft_number):
    if pd.isna(draft_number):
        return 0.02
    if draft_number <= 10:
        return 0.95 - ((draft_number - 1) / 9) * 0.10
    if draft_number <= 32:
        return 0.80 - ((draft_number - 11) / 21) * 0.25
    return max(0.25 - (draft_number / 300), 0.02)


allpos_active.loc[allpos_active['is_rookie'], 'competition_quality'] = allpos_active.loc[
    allpos_active['is_rookie'], 'draft_number'].apply(rookie_quality)
allpos_active['current_team_pos_rank'] = allpos_active.groupby('team')['competition_quality'].rank(ascending=False, method='min')

wr_active = allpos_active[allpos_active['position'] == 'WR'].rename(columns={'player_id': 'player_id'})

season = season.merge(
    wr_active[['player_id', 'current_team_pos_rank', 'team']].rename(columns={'team': f'team_{PREDICTION_SEASON}'}),
    on='player_id', how='left')

# ============================================================
# STEP D: Final shortlist
# ============================================================
result = season.dropna(subset=['current_team_pos_rank'])
result = result[result['pos_rank'] > 12]  # exclude already-elite players
shortlist = result[result['current_team_pos_rank'] <= 2].sort_values('breakout_probability', ascending=False)
shortlist = shortlist.sort_values('target_share', ascending=False).drop_duplicates(subset=['player_display_name'], keep='first')
shortlist = shortlist.sort_values('breakout_probability', ascending=False)

cols = ['player_display_name', f'team_{PREDICTION_SEASON}', 'current_team_pos_rank', 'ppr_ppg',
        'target_share', 'ppr_per_target', 'breakout_probability']
shortlist[cols].head(25).to_csv('../predictions/wr_2026_shortlist.csv', index=False)
print(shortlist[cols].head(25).to_string(index=False))
