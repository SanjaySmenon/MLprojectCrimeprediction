# =============================
# 1️⃣ Import Libraries
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# =============================
# 2️⃣ Load Dataset
# =============================
file_path = r"D:\SANJAY\MTECH\ML project\CrimeDataBD\crime_dataset_india.csv"
df = pd.read_csv(file_path)

# Keep relevant columns
df = df[['Date of Occurrence', 'Time of Occurrence', 'City', 'Victim Age',
         'Victim Gender', 'Weapon Used', 'Crime Domain', 'Police Deployed']]

# =============================
# 3️⃣ Preprocess Data
# =============================

# Convert date/time columns
df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'], format='%d-%m-%Y %H:%M', errors='coerce')
df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], format='%d-%m-%Y %H:%M', errors='coerce')

# Extract features
df['Occur_Year'] = df['Date of Occurrence'].dt.year
df['Occur_Month'] = df['Date of Occurrence'].dt.month
df['Occur_Day'] = df['Date of Occurrence'].dt.day
df['Occur_Weekday'] = df['Date of Occurrence'].dt.weekday  # Monday=0
df['Occur_Hour'] = df['Time of Occurrence'].dt.hour
df['Occur_Minute'] = df['Time of Occurrence'].dt.minute

# Drop original date/time
df.drop(columns=['Date of Occurrence', 'Time of Occurrence'], inplace=True)

# Handle missing values
numeric_cols = ['Victim Age', 'Police Deployed', 'Occur_Year', 'Occur_Month',
                'Occur_Day', 'Occur_Weekday', 'Occur_Hour', 'Occur_Minute']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = ['City', 'Victim Gender', 'Weapon Used', 'Crime Domain']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical features
le_city = LabelEncoder()
le_gender = LabelEncoder()
le_weapon = LabelEncoder()
le_target = LabelEncoder()  # Crime Domain

df['City'] = le_city.fit_transform(df['City'])
df['Victim Gender'] = le_gender.fit_transform(df['Victim Gender'])
df['Weapon Used'] = le_weapon.fit_transform(df['Weapon Used'])
df['Crime Domain'] = le_target.fit_transform(df['Crime Domain'])

# =============================
# 4️⃣ Prepare Features and Target
# =============================
X = df[['City', 'Victim Age', 'Victim Gender', 'Weapon Used', 'Police Deployed',
        'Occur_Year', 'Occur_Month', 'Occur_Day', 'Occur_Weekday', 'Occur_Hour', 'Occur_Minute']]
y = df['Crime Domain']  # Target

# --- Visualize class imbalance before SMOTE ---
plt.figure(figsize=(8, 4))
sns.countplot(x=y)
plt.title("Class Distribution Before SMOTE")
plt.xlabel("Crime Domain")
plt.ylabel("Count")
plt.show()

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# --- Visualize class balance after SMOTE ---
plt.figure(figsize=(8, 4))
sns.countplot(x=y_res)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Crime Domain")
plt.ylabel("Count")
plt.show()

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.1, random_state=42)

# =============================
# 5️⃣ Train Random Forest Classifier
# =============================
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# =============================
# 6️⃣ Make Predictions
# =============================
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

# =============================
# 7️⃣ Evaluate Model
# =============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# --- Feature Importance Visualization ---
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# =============================
# 8️⃣ Predict on New Input Example
# =============================
new_data = pd.DataFrame({
    'City': le_city.transform(['Pune']),
    'Victim Age': [25],
    'Victim Gender': le_gender.transform(['F']),
    'Weapon Used': le_weapon.transform(['Blunt Object']),
    'Police Deployed': [5],
    'Occur_Year': [2025],
    'Occur_Month': [10],
    'Occur_Day': [3],
    'Occur_Weekday': [3],  # Example: Thursday
    'Occur_Hour': [14],
    'Occur_Minute': [30]
})

pred_class = rf.predict(new_data)
pred_prob = rf.predict_proba(new_data)

print("\nPredicted Crime Domain:", le_target.inverse_transform(pred_class)[0])
print("Prediction Probabilities for all classes:", pred_prob)
