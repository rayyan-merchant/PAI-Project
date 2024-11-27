import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.combine import SMOTETomek
import joblib

try:
    dataset = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please ensure the file is in the correct directory.")
    exit()

print("\nInitial Data Inspection:")
print(dataset.head())
print(dataset.info())
print(dataset.describe())

# Data preprocessing
dataset.fillna(dataset.median(), inplace=True)
dataset['BMI'] = dataset['BMI'].apply(lambda x: dataset['BMI'].median() if x < 0 else x)
dataset['Age'] = dataset['Age'].apply(lambda x: dataset['Age'].median() if x < 0 else x)

for column in dataset.select_dtypes(include=np.number).columns:
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset[column] = np.clip(dataset[column], lower_bound, upper_bound)

    if 'Outcome' not in dataset.columns:
        print("Error: 'Outcome' column not found. Ensure the dataset contains a target variable.")
        exit()

X = dataset.drop(columns=['Outcome'])
y = dataset['Outcome']

# Handle class imbalance with SMOTETomek
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

# VotingClassifier
voting_clf = VotingClassifier( estimators=[ ('RandomForest', rf), ('KNN', knn) ], voting='soft' )
voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)
y_pred_prob = voting_clf.predict_proba(X_test)[:, 1]

# Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
print(f"\nAccuracy: {accuracy}")
print(f"ROC-AUC Score: {roc_auc}")


joblib.dump(voting_clf, 'diabetes_model.pkl')
print("Ensemble model saved as diabetes_model.pkl")