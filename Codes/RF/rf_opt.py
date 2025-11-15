from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load dataset
X=np.load('/rds/projects/g/grynovag-chem-space/transfer_2/NN_MRS/google_sound/X.npy')
y=np.load('/rds/projects/g/grynovag-chem-space/transfer_2/NN_MRS/google_sound/Y.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=137, stratify=y
)

# Hyperparameter search space
param_dist = {
    'n_estimators': [200],
    'max_depth': [40],
    'min_samples_split': [5],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'bootstrap': [False]
}

# Optimize Random Forest
rf = RandomForestClassifier(random_state=137,n_jobs=-1)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=137,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

# Evaluate optimized model
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Generate confusion matrix and save as NumPy file
cm = confusion_matrix(y_test, y_pred)


