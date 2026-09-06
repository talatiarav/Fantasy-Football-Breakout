"""
Position-specific models, restricted to the NGS-covered population only (no
median imputation -- every row has real tracking data). Includes both
opportunity/volume features AND niche NGS features together, per the idea
that volume shouldn't be dropped just because we're adding tracking stats.

TE is excluded entirely: only 3 total breakout examples in the TE-covered
population (before even splitting train/test) -- not enough to train or
evaluate anything meaningful. RB has very few positives too (6 train / 3
test) and should be read as illustrative, not reliable. WR has real,
if still modest, sample size (45 train / 9 test positives).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv('training_table.csv')

opportunity_cols = [
    'target_share', 'air_yards_share', 'wopr', 'racr', 'avg_snap_pct',
    'age', 'years_exp', 'draft_number', 'games_played', 'ppr_ppg', 'pos_rank',
    'receiving_air_yards', 'receiving_yards_after_catch',
]
wr_te_ngs_cols = ['avg_cushion', 'avg_separation', 'percent_share_of_intended_air_yards',
                  'avg_yac_above_expectation', 'qb_cpoe']
rb_extra_cols = ['carries', 'rushing_yards']
rb_ngs_cols = ['efficiency', 'percent_attempts_gte_eight_defenders',
               'rush_yards_over_expected_per_att']


def run_position_model(position, ngs_flag_col, extra_opportunity_cols, ngs_cols, model_type='logreg'):
    sub = df[df['position'] == position].copy()
    covered = sub[sub[ngs_flag_col] == 1].copy()
    feature_cols = opportunity_cols + extra_opportunity_cols + ngs_cols
    covered = covered.dropna(subset=feature_cols)  # covered population should have full coverage;
                                                     # drop the rare remaining gap rather than impute

    train_mask = covered['season_n'] <= 2021
    X = covered[feature_cols]
    y = covered['breakout']
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"\n=== {position} (NGS-covered only) ===")
    print(f"Train: {len(X_train)} rows, {y_train.sum()} breakouts | Test: {len(X_test)} rows, {y_test.sum()} breakouts")

    if y_train.sum() < 5 or y_test.sum() < 3:
        print("TOO FEW POSITIVE EXAMPLES -- results below are illustrative only, not reliable.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'logreg':
        model = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
    else:
        model = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    prob = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"{position}-specific model ROC AUC: {auc:.4f}")

    if model_type == 'logreg':
        importances = pd.Series(np.abs(model.coef_[0]), index=feature_cols).sort_values(ascending=False)
    else:
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"Top features:\n{importances.head(8)}")

    return {'position': position, 'auc': auc, 'n_train': len(X_train), 'n_test': len(X_test),
            'train_positives': int(y_train.sum()), 'test_positives': int(y_test.sum()),
            'importances': importances}


results = []
results.append(run_position_model('WR', 'has_ngs_receiving', [], wr_te_ngs_cols, model_type='logreg'))
results.append(run_position_model('RB', 'has_ngs_rushing', rb_extra_cols, rb_ngs_cols, model_type='logreg'))

# --- TE: not modeled, but show why ---
te_covered = df[(df['position'] == 'TE') & (df['has_ngs_receiving'] == 1)]
print(f"\n=== TE (NGS-covered only) ===")
print(f"Total covered rows: {len(te_covered)}, total breakouts: {te_covered['breakout'].sum()}")
print("Not modeled -- too few positive examples to train or evaluate anything meaningful.")

# --- Compare against the general combined model, evaluated on these SAME covered subsets ---
print("\n=== For comparison: general box-score-only model, evaluated on the same covered subsets ===")
import joblib
general = joblib.load('breakout_model_final.joblib')
gen_model, gen_scaler, gen_cols = general['model'], general['scaler'], general['feature_cols']

df_dummies = pd.get_dummies(df, columns=['position'], prefix='pos')
df_dummies['avg_snap_pct'] = df_dummies['avg_snap_pct'].fillna(df_dummies['avg_snap_pct'].median())
df_dummies['racr'] = df_dummies['racr'].fillna(df_dummies['racr'].median())

for position, flag_col in [('WR', 'has_ngs_receiving'), ('RB', 'has_ngs_rushing')]:
    pos_col = f'pos_{position}'
    covered_mask = (df_dummies[pos_col] == True) & (df_dummies[flag_col] == 1) & (df_dummies['season_n'] > 2021)
    covered_test = df_dummies[covered_mask]
    if len(covered_test) == 0 or covered_test['breakout'].sum() == 0:
        print(f"{position}: not enough test positives to compare")
        continue
    X_gen = covered_test[gen_cols]
    X_gen_scaled = gen_scaler.transform(X_gen)
    gen_prob = gen_model.predict_proba(X_gen_scaled)[:, 1]
    gen_auc = roc_auc_score(covered_test['breakout'], gen_prob)
    print(f"{position}: general model ROC AUC on this exact covered test subset: {gen_auc:.4f}")
