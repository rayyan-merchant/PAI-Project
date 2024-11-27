import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
from collections import Counter
import joblib


def load_dataset():
    try:
        dataset = pd.read_csv("/content/diabetes.csv")
        print("Dataset loaded successfully!")
        return dataset
    except FileNotFoundError:
        print(f"Error: diabetes.csv not found. Please ensure the file is in the correct directory.")
        exit()


def clean_and_preprocess_data(dataset):
    dataset.fillna(dataset.median(), inplace=True)

    columns_to_fix = ['BMI', 'Age']
    for column in columns_to_fix:
        dataset[column] = dataset[column].apply(lambda x: dataset[column].median() if x < 0 else x)

    for column in dataset.select_dtypes(include=np.number).columns:
        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset[column] = np.clip(dataset[column], lower_bound, upper_bound)

    return dataset


def perform_eda(dataset):
    for column in dataset.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(dataset[column], kde=True, bins=30, color='pink')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    plt.figure(figsize=(15, 8))
    sns.boxplot(data=dataset, palette="Set3")
    plt.title("Boxplot for All Features")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

    sns.pairplot(dataset, hue="Outcome", palette="husl", diag_kind="kde")
    plt.suptitle("Pairplot of Features by Outcome", y=1.02)
    plt.show()


def analyze_class_imbalance(y_before, y_after):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.countplot(x=y_before)
    plt.title('Class Distribution Before SMOTETomek')
    plt.xlabel('Outcome')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.countplot(x=y_after)
    plt.title('Class Distribution After SMOTETomek')
    plt.xlabel('Outcome')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    print("\nClass distribution before SMOTETomek:", Counter(y_before))
    print("Class distribution after SMOTETomek:", Counter(y_after))


def plot_roc_curve(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


def plot_classification_report(report, title):
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title(title)
    plt.show()


def main():
    dataset = load_dataset()
    print("Initial dataset shape:", dataset.shape)
    print(dataset.info())
    print(dataset.describe())

    dataset = clean_and_preprocess_data(dataset)

    perform_eda(dataset)

    X = dataset.drop(columns=['Outcome'])
    y = dataset['Outcome']

    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)

    analyze_class_imbalance(y, y_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                        stratify=y_resampled)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

    rf_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_cm = confusion_matrix(y_test, rf_pred)
    rf_report = classification_report(y_test, rf_pred, output_dict=True)

    knn_pred = knn_model.predict(X_test)
    knn_pred_proba = knn_model.predict_proba(X_test)[:, 1]
    knn_cm = confusion_matrix(y_test, knn_pred)
    knn_report = classification_report(y_test, knn_pred, output_dict=True)

    print("\nRandom Forest Metrics:")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, rf_pred_proba))
    print("Classification Report:\n", classification_report(y_test, rf_pred))

    print("\nKNN Metrics:")
    print("Accuracy:", accuracy_score(y_test, knn_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, knn_pred_proba))
    print("Classification Report:\n", classification_report(y_test, knn_pred))

    plot_confusion_matrix(rf_cm, "Random Forest Confusion Matrix")
    plot_confusion_matrix(knn_cm, "KNN Confusion Matrix")
    plot_classification_report(rf_report, "Random Forest Classification Report")
    plot_classification_report(knn_report, "KNN Classification Report")

    voting_clf = VotingClassifier(estimators=[('RandomForest', rf_model), ('KNN', knn_model)], voting='soft')

    voting_clf.fit(X_train, y_train)

    y_pred = voting_clf.predict(X_test)
    y_pred_prob = voting_clf.predict_proba(X_test)[:, 1]

    ensemble_cm = confusion_matrix(y_test, y_pred)
    ensemble_report = classification_report(y_test, y_pred, output_dict=True)
    ensemble_accuracy = accuracy_score(y_test, y_pred)
    ensemble_roc_auc = roc_auc_score(y_test, y_pred_prob)

    print("\nEnsemble Model Metrics:")
    print("Accuracy:", ensemble_accuracy)
    print("ROC-AUC Score:", ensemble_roc_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    plot_confusion_matrix(ensemble_cm, "Ensemble Voting Classifier Confusion Matrix")
    plot_classification_report(ensemble_report, "Ensemble Voting Classifier Classification Report")

    plt.figure(figsize=(10, 6))
    plot_roc_curve(y_test, rf_pred_proba, 'Random Forest')
    plot_roc_curve(y_test, knn_pred_proba, 'KNN')
    plot_roc_curve(y_test, y_pred_prob, 'Voting Classifier')
    plt.legend(loc='lower right')
    plt.show()

    rf_feature_importance = rf_model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_feature_importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(knn_model, 'knn_model.pkl')
    joblib.dump(voting_clf, 'voting_classifier_model.pkl')
    print("Models saved: 'random_forest_model.pkl', 'knn_model.pkl', and 'voting_classifier_model.pkl'")


if __name__ == "__main__":
    main()
