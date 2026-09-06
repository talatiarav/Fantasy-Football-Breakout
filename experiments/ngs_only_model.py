"""
NGS-only model: train exclusively on the niche tracking stats (separation,
cushion, YAC over expectation, RYOE, QB CPOE, etc.) -- no target share, no
PPR points, no draft capital, no age. The question: how much predictive
power do pure "underlying skill/efficiency" metrics carry on their own,
independent of how much a player is actually being used?

Restricted to the NGS-covered population only (players above the volume
threshold), since these features don't exist otherwise.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv('training_table.csv')
df = pd.get_dummies(df, columns=['position'], prefix='pos')
df['has_ngs_coverage'] = ((df['has_ngs_receiving'] == 1) | (df['has_ngs_rushing'] == 1)).astype(int)

covered = df[df['has_ngs_coverage'] == 1].copy()

# Pure NGS/tracking features only -- deliberately excluding target_share, ppr_ppg,
# pos_rank, draft_number, age, snap_pct, etc. so the model has NO opportunity/volume
# signal to lean on, only efficiency/skill metrics.
ngs_only_cols = [
    'avg_cushion', 'avg_separation', 'percent_share_of_intended_air_yards', 'avg_yac_above_expectation',
    'efficiency', 'percent_attempts_gte_eight_defenders', 'rush_yards_over_expected_per_att',
    'qb_cpoe', 'qb_time_to_throw', 'qb_aggressiveness',
    'pos_RB', 'pos_TE', 'pos_WR',  # position still matters for interpreting these stats
]

train_mask = covered['season_n'] <= 2021
X = covered[ngs_only_cols]
y = covered['breakout']
X_train, X_test = X[train_mask].copy(), X[~train_mask].copy()
y_train, y_test = y[train_mask], y[~train_mask]

print(f"NGS-covered population: {len(covered)} ({train_mask.sum()} train / {(~train_mask).sum()} test)")
print(f"Breakout rate -- train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# Fill position-specific gaps (WR won't have RB rushing stats, etc.) with train-only median
for col in ngs_only_cols:
    if X_train[col].isna().any() or X_test[col].isna().any():
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
logreg_prob = logreg.predict_proba(X_test_scaled)[:, 1]
print(f"\nLogistic Regression (NGS-only) ROC AUC: {roc_auc_score(y_test, logreg_prob):.4f}")

gbc = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42)
gbc.fit(X_train_scaled, y_train)
gbc_prob = gbc.predict_proba(X_test_scaled)[:, 1]
print(f"Gradient Boosting (NGS-only) ROC AUC:   {roc_auc_score(y_test, gbc_prob):.4f}")

importances = pd.Series(logreg.coef_[0], index=ngs_only_cols).sort_values(key=abs, ascending=False)
print("\nLogistic regression coefficients (sign shows direction, magnitude shows strength):")
print(importances)

# For reference: how well does the box-score model do on this SAME covered subset?
# (pulled from the two-stage experiment result: 0.8066 ROC AUC on covered test rows)
print(f"\nFor comparison, box-score-only model on this same covered test subset: 0.8066 ROC AUC")
