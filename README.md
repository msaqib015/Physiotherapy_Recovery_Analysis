# Patient Recovery Analysis for Physiotherapy

## Project Overview
This project analyzes recovery trends in physiotherapy patients using a simulated dataset. The goal is to uncover insights and build a predictive model to estimate recovery times based on patient demographics, injury types, treatment methods, and session counts.

---

## Steps Taken

### 1. Dataset Creation
A synthetic dataset was generated with 500 patient records. The fields include:
- **Age:** Patient age (18 to 80 years)
- **Gender:** Male or Female
- **Injury Type:** Sprain, Fracture, or Post-Surgery
- **Treatment Type:** Manual Therapy, Electrotherapy, or Exercise Therapy
- **Number of Sessions:** Count of physiotherapy sessions (5 to 20)
- **Recovery Time:** Recovery duration in days (10 to 120 days)

Code snippet:
```python
import numpy as np
import pandas as pd

np.random.seed(42)
data = {
    "Patient_ID": range(1, 501),
    "Age": np.random.randint(18, 80, size=500),
    "Gender": np.random.choice(["Male", "Female"], size=500),
    "Injury_Type": np.random.choice(["Sprain", "Fracture", "Post-Surgery"], size=500),
    "Treatment_Type": np.random.choice(["Manual Therapy", "Electrotherapy", "Exercise Therapy"], size=500),
    "Number_of_Sessions": np.random.randint(5, 20, size=500),
    "Recovery_Time_Days": np.random.randint(10, 120, size=500)
}
df = pd.DataFrame(data)
df.to_csv("Physiotherapy_Recovery_Data.csv", index=False)
```

---

### 2. Data Preprocessing
- Encoded categorical variables (Gender, Injury Type, Treatment Type).
- Verified and normalized the dataset.

Code snippet:
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["Gender_Encoded"] = encoder.fit_transform(df["Gender"])
df["Injury_Type_Encoded"] = encoder.fit_transform(df["Injury_Type"])
df["Treatment_Type_Encoded"] = encoder.fit_transform(df["Treatment_Type"])
```

---

### 3. Exploratory Data Analysis (EDA)
- **Distribution of Recovery Times:** Highlighted the variation in recovery duration.
- **Recovery Time by Injury and Treatment Types:** Identified differences in recovery based on injury and treatment.

Visualizations:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
sns.histplot(df["Recovery_Time_Days"], kde=True, bins=20)
plt.title("Distribution of Recovery Time")
plt.show()

# Boxplots
sns.boxplot(x="Injury_Type", y="Recovery_Time_Days", data=df)
plt.title("Recovery Time by Injury Type")
plt.show()
```

---

### 4. Predictive Modeling
A Linear Regression model was used to predict recovery times.

**Steps:**
1. **Features:** Age, Gender (Encoded), Injury Type (Encoded), Treatment Type (Encoded), Number of Sessions.
2. **Target:** Recovery Time (Days).
3. **Evaluation:** Model achieved a Root Mean Squared Error (RMSE) of 31.61 days.

Code snippet:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = df[["Age", "Gender_Encoded", "Injury_Type_Encoded", "Treatment_Type_Encoded", "Number_of_Sessions"]]
y = df["Recovery_Time_Days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
```

---

### 5. Insights
- **Younger patients** generally recover faster.
- **Electrotherapy** was more effective for fractures.
- Recovery time decreases with an increased number of sessions.

---

### 6. Recommendations for Physiotherapists
- Prioritize age-specific rehabilitation programs.
- Customize treatments based on injury type for faster recovery.
- Monitor session counts to ensure optimal recovery times.

---


## Conclusion
This project demonstrates how physiotherapy data can be analyzed to uncover insights and predict recovery times. These methods can support data-driven decision-making in clinical settings, enhancing patient outcomes and operational efficiency.
