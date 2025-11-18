# ================================
# Adversarial Validation with CatBoost + Feature Importance
# ================================

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# --------------------------------
# 1. Combine train and test
# --------------------------------

train_df_av = train_df.copy()
test_df_av = test_df.copy()

train_df_av["is_test"] = 0
test_df_av["is_test"] = 1

full = pd.concat([train_df_av, test_df_av], ignore_index=True)

target = full["is_test"]
features = full.drop(columns=["is_test", "target"], errors="ignore")

cat_features = features.select_dtypes(include=["object", "category"]).columns.tolist()

# --------------------------------
# 2. Train adversarial model
# --------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0
)

model.fit(
    Pool(X_train, y_train, cat_features=cat_features),
    eval_set=Pool(X_valid, y_valid, cat_features=cat_features)
)

# --------------------------------
# 3. AUC score — the main AV metric
# --------------------------------
valid_pred = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, valid_pred)
print(f"AUC of adversarial classifier: {auc:.4f}")

# --------------------------------
# 4. Feature importance (most important = most distribution-shifted)
# --------------------------------
importances = model.get_feature_importance(Pool(features, target, cat_features=cat_features))
feat_imp = pd.DataFrame({
    "feature": features.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop train-vs-test drifting features:")
display(feat_imp.head(20))

# --------------------------------
# 5. Get "probability of being test" for each row
# --------------------------------
probs_test_like = model.predict_proba(features)[:, 1]

train_df["av_score"] = probs_test_like[full["is_test"] == 0]

# Optional: see most suspicious rows
train_df.sort_values("av_score", ascending=False).head()

