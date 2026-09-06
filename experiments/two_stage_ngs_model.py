"""
Two-stage breakout model.

Stage 1 (base): box-score features only, trained on the FULL population
(2013-2024, ~2000 training rows). This is the model that achieved 0.7851
ROC AUC on its own -- full feature coverage, no missingness to dilute it.

Stage 2 (refinement): for players who clear the NGS volume threshold
(has_ngs_receiving or has_ngs_rushing == 1), train a second model using
NGS tracking features PLUS the Stage 1 predicted probability as an input.
This only trains on players who actually have real tracking data, so there's
no median-imputation noise -- the refinement either helps in the region
where it has real signal, or it doesn't, and we can measure that directly.

Final prediction: Stage 2 output for covered players, Stage 1 output for
everyone else.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv('training_table.csv')
df = df.dropna(subset=['target_share', 'air_yards_share', 'wopr', 'racr'])
df = pd.get_dummies(df, columns=['position'], prefix='pos')

box_score_cols = [
    'target_share', 'air_yards_share', 'wopr', 'racr', 'avg_snap_pct',
    'age', 'years_exp', 'draft_number', 'games_played', 'ppr_ppg', 'pos_rank',
    'receiving_air_yards', 'receiving_yards_after_catch', 'carries', 'rushing_yards',
    'pos_RB', 'pos_TE', 'pos_WR',
]
ngs_cols = [
    'avg_cushion', 'avg_separation', 'percent_share_of_intended_air_yards', 'avg_yac_above_expectation',
    'efficiency', 'percent_attempts_gte_eight_defenders', 'rush_yards_over_expected_per_att',
    'qb_cpoe', 'qb_time_to_throw', 'qb_aggressiveness',
]

df['avg_snap_pct'] = df['avg_snap_pct'].fillna(df.loc[df['season_n'] <= 2021, 'avg_snap_pct'].median())
df['has_ngs_coverage'] = ((df['has_ngs_receiving'] == 1) | (df['has_ngs_rushing'] == 1)).astype(int)

train_mask = df['season_n'] <= 2021
y = df['breakout']

# ============ STAGE 1: base model, full population, box-score features ============
X_box = df[box_score_cols]
X_box_train, X_box_test = X_box[train_mask], X_box[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

scaler1 = StandardScaler()
X_box_train_scaled = scaler1.fit_transform(X_box_train)
X_box_test_scaled = scaler1.transform(X_box_test)

stage1 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
stage1.fit(X_box_train_scaled, y_train)

df.loc[train_mask, 'stage1_prob'] = stage1.predict_proba(X_box_train_scaled)[:, 1]
df.loc[~train_mask, 'stage1_prob'] = stage1.predict_proba(X_box_test_scaled)[:, 1]

stage1_auc_full = roc_auc_score(y_test, df.loc[~train_mask, 'stage1_prob'])
print(f"Stage 1 (base, full population) ROC AUC on full test set: {stage1_auc_full:.4f}")

# ============ STAGE 2: refinement model, NGS-covered players only ============
covered = df[df['has_ngs_coverage'] == 1].copy()
# Trim to the NGS features that showed real importance in the earlier all-features model
# (avg_yac_above_expectation, avg_separation, avg_cushion, qb_cpoe) rather than all 10 --
# with ~800 training rows and ~53 positive examples, every extra feature is a real cost.
top_ngs_cols = ['avg_yac_above_expectation', 'avg_separation', 'avg_cushion', 'qb_cpoe']
stage2_features = top_ngs_cols + ['stage1_prob']

cov_train_mask = covered['season_n'] <= 2021
X_cov_train = covered.loc[cov_train_mask, stage2_features]
X_cov_test = covered.loc[~cov_train_mask, stage2_features]
y_cov_train = covered.loc[cov_train_mask, 'breakout']
y_cov_test = covered.loc[~cov_train_mask, 'breakout']

# Within "covered," a WR still won't have rushing NGS values (and an RB won't have
# receiving NGS values) -- those are legitimately position-specific gaps, not a coverage
# issue. Fill with the train-covered median, same no-leakage discipline as Stage 1.
X_cov_train = X_cov_train.copy()
X_cov_test = X_cov_test.copy()
for col in stage2_features:
    if X_cov_train[col].isna().any() or X_cov_test[col].isna().any():
        med = X_cov_train[col].median()
        X_cov_train[col] = X_cov_train[col].fillna(med)
        X_cov_test[col] = X_cov_test[col].fillna(med)

print(f"\nNGS-covered population: {len(covered)} total ({cov_train_mask.sum()} train / {(~cov_train_mask).sum()} test)")
print(f"Covered breakout rate -- train: {y_cov_train.mean():.3f}, test: {y_cov_test.mean():.3f}")

scaler2 = StandardScaler()
X_cov_train_scaled = scaler2.fit_transform(X_cov_train)
X_cov_test_scaled = scaler2.transform(X_cov_test)

# Smaller, more regularized model -- the covered subset is a fraction of the full training
# set, so a large GBC risks overfitting. Logistic regression with L2 is a safer refinement layer.
stage2 = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.2)
stage2.fit(X_cov_train_scaled, y_cov_train)
stage2_prob_test = stage2.predict_proba(X_cov_test_scaled)[:, 1]

stage2_auc_covered = roc_auc_score(y_cov_test, stage2_prob_test)
stage1_auc_covered = roc_auc_score(y_cov_test, covered.loc[~cov_train_mask, 'stage1_prob'])
print(f"\nOn the NGS-covered test subset only:")
print(f"  Stage 1 alone (box-score) ROC AUC: {stage1_auc_covered:.4f}")
print(f"  Stage 2 (+ NGS refinement) ROC AUC: {stage2_auc_covered:.4f}")

# ============ Blended final prediction: Stage 2 for covered, Stage 1 for everyone else ============
df['final_prob'] = df['stage1_prob']
covered_test_idx = covered.loc[~cov_train_mask].index
df.loc[covered_test_idx, 'final_prob'] = stage2_prob_test

blended_auc_full = roc_auc_score(y_test, df.loc[~train_mask, 'final_prob'])
print(f"\nBlended model (Stage 2 where covered, Stage 1 elsewhere) ROC AUC on full test set: {blended_auc_full:.4f}")
print(f"(vs. Stage 1 alone on full test set: {stage1_auc_full:.4f})")

import joblib
joblib.dump({
    'stage1_model': stage1, 'stage1_scaler': scaler1, 'box_score_cols': box_score_cols,
    'stage2_model': stage2, 'stage2_scaler': scaler2, 'stage2_features': stage2_features,
}, 'breakout_model_two_stage.joblib')
print("\nSaved breakout_model_two_stage.joblib")
