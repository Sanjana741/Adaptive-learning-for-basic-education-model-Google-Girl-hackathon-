# Knowledge Difficulty Model Training (Scikit-learn, TensorFlow/PyTorch):
# Choose a suitable machine learning algorithm (e.g., Random Forest) or deep learning architecture (if applicable).
# Train the model using the preprocessed data, with features as input and difficulty score as output.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create a pandas DataFrame from the data dictionary
data = {
    "Student ID": [647, 41, 340, 641, 669, 697, 720, 675, 663, 28, 953, 636, 639],
    "Student Country": ["Ireland", "Portugal", "Portugal", "Italy", "Portugal", "Portugal", "Portugal", "Portugal", "Portugal", "Portugal", "Lithuania", "Italy", "Italy"],
    "Question ID": [77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77],
    "Type of Answer": [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    "Question Level": ["Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic"],
    "Topic": ["Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics"],
    "Subtopic": ["Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics"],
    "Keywords": ["Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency", "Stem and Leaf diagram Relative frequency  Sample  Frequency"]
}

df = pd.DataFrame(data)

# **Handle categorical features (e.g., Student Country, Question Level)**
categorical_features = ['Student Country', 'Question Level', 'Topic', 'Subtopic']

# One-hot encode categorical features
df = pd.get_dummies(df, columns=categorical_features)

# **Drop the "Keywords" feature for now**
# Explore text preprocessing and feature extraction for a more comprehensive approach later
df.drop('Keywords', axis=1, inplace=True)

# Extract features (independent variables) and target variable (dependent variable)
X = df.drop('Type of Answer', axis=1)  # Features
y = df['Type of Answer']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set (optional)
# You can use this to evaluate model performance on unseen data
y_pred = model.predict(X_test)

# Evaluate model performance (optional)
# Use metrics like accuracy, precision, recall, F1-score, etc.
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")

# Save the model (optional)
# from sklearn.externals import joblib
# joblib.dump(model, 'knowledge_difficulty_model.pkl')

print("Model training complete!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Knowledge Mastery Model (Rule-based system):
# Develop a rule-based system that considers factors like normalized exercise scores and video watching behavior to classify student mastery levels (unlearned, unmastered, etc.)
def classify_mastery(nes, threshold_unlearned, threshold_unmastered, threshold_mastered):
  """
  Classifies student mastery level based on normalized exercise score (NES).

  Args:
    nes: Normalized exercise score (0-1).
    threshold_unlearned: Threshold for "Unlearned" level.
    threshold_unmastered: Threshold for "Unmastered" level.
    threshold_mastered: Threshold for "Mastered" level.

  Returns:
    A string representing the mastery level ("Unlearned", "Unmastered",
                                           "Insufficiently Mastered", "Mastered").
  """
  if nes < threshold_unlearned:
    return "Unlearned"
  elif nes < threshold_unmastered:
    return "Unmastered"
  elif nes < threshold_mastered:
    return "Insufficiently Mastered"
  else:
    return "Mastered"

# Example usage
nes = 0.75
threshold_unlearned = 0.2
threshold_unmastered = 0.6
threshold_mastered = 0.85

mastery_level = classify_mastery(nes, threshold_unlearned, threshold_unmastered, threshold_mastered)
print("Mastery Level:", mastery_level)


