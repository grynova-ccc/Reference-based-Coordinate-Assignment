import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
X_audio = np.load('PATH/X.npy')
X_space = np.load('PATH/RCA_30.npy')
y = np.load('PATH/Y.npy')

X_audio_train, X_audio_test, X_space_train, X_space_test, y_train, y_test = train_test_split(
    X_audio, X_space, y, test_size=0.2, random_state=42, stratify=y
)

# Parameter grid for Random Forests
rf_param_dist1 = {
    'n_estimators': [200],
    'max_depth': [40],
    'min_samples_split': [5],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'bootstrap': [False]
}

rf_param_dist2 = {
    'n_estimators': [500],
    'max_depth': [40],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['log2'],
    'bootstrap': [True]
}

        
# Optimize RF for audio features
rf_audio = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_audio_search = RandomizedSearchCV(
    rf_audio,
    rf_param_dist1,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rf_audio_search.fit(X_audio_train, y_train)
best_rf_audio = rf_audio_search.best_estimator_

# Optimize RF for space features
rf_space = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_space_search = RandomizedSearchCV(
    rf_space,
    rf_param_dist2,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
rf_space_search.fit(X_space_train, y_train)
best_rf_space = rf_space_search.best_estimator_

# Generate meta-features (predicted probabilities)
proba_audio_train = best_rf_audio.predict_proba(X_audio_train)
proba_audio_test = best_rf_audio.predict_proba(X_audio_test)

proba_space_train = best_rf_space.predict_proba(X_space_train)
proba_space_test = best_rf_space.predict_proba(X_space_test)

X_meta_train = np.hstack([proba_audio_train, proba_space_train])
X_meta_test = np.hstack([proba_audio_test, proba_space_test])

# Optimize Logistic Regression meta-classifier
meta_param_dist = {
    'C': [np.float64(0.001)],
    'solver': ['lbfgs'],
    'multi_class': ['multinomial']
}

meta_clf = LogisticRegression(max_iter=1000)
meta_search = RandomizedSearchCV(
    meta_clf,
    meta_param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
meta_search.fit(X_meta_train, y_train)
best_meta_clf = meta_search.best_estimator_

# Final predictions and evaluation
y_pred = best_meta_clf.predict(X_meta_test)

print("Best RF (Full):", rf_audio_search.best_params_)
print("Best RF (RCA):", rf_space_search.best_params_)
print("Best Meta-Classifier:", meta_search.best_params_)
print("Stacked Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
