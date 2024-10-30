import matplotlib.pyplot as plt
import pandas as pd

a = pd.read_csv('diabetes.csv')
print(a)
outcome_counts = a['Outcome'].value_counts()
label= ["Non Diabetic","Diabetic"]
plt.pie(outcome_counts, labels=label, autopct='%1.1f%%',colors = ["pink","lightblue"])
plt.title('Diabetes Outcomes')
plt.axis('equal')
plt.show()

a.hist(bins=10, figsize=(15, 10), layout=(3, 3))
plt.tight_layout()
plt.show()
plt.figure(figsize=(15, 10))
for i, column in enumerate(a.columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(a[column])
    plt.title(column)
plt.tight_layout()
plt.show()

plt.scatter(a['Pregnancies'], a['Age'],label='Data Points')
plt.xlabel('Number of Pregnancies')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. Pregnancies')
plt.legend()
plt.show()


plt.scatter(a['Glucose'],a['Age'],label='Data Points')
plt.xlabel('Glucose')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. Glucose')
plt.legend()
plt.show()

plt.scatter(a['BloodPressure'],a['Age'],label='Data Points')
plt.xlabel('BloodPressure')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. BloodPressure')
plt.legend()
plt.show()

plt.scatter(a['SkinThickness'],a['Age'],label='Data Points')
plt.xlabel('SkinThickness')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. SkinThickness')
plt.legend()
plt.show()

plt.scatter(a['Insulin'],a['Age'],label='Data Points')
plt.xlabel('Insulin')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. Insulin')
plt.legend()
plt.show()  

plt.scatter(a['BMI'],a['Age'],label='Data Points')
plt.xlabel('BMI')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. BMI')
plt.legend()
plt.show()

plt.scatter(a['DiabetesPedigreeFunction'],a['Age'],label='Data Points')
plt.xlabel('DiabetesPedigreeFunction')
plt.ylabel('Age')
plt.title('Scatter Plot of Age vs. DiabetesPedigreeFunction')
plt.legend()
plt.show()

plt.scatter(a['Age'],a['Outcome'],label='Data Points')
plt.xlabel('Age')
plt.ylabel('Outcome')
plt.title('Scatter Plot of Age vs. Outcome')
plt.legend()
plt.show()
