# Load student data (exercise scores, video-watching behavior, comments) and course content (knowledge points, prerequisite relationships).
# Clean and pre-process the data (handle missing values, normalize scores).
# Extract features for the Knowledge Difficulty Model (e.g., average score, repetitions, comments) for each knowledge point.
import pandas as pd

# Sample data (replace with your actual data)
data = {
    "Student ID": [647, 41, 340, 641, 669, 697, 720, 675, 663, 28, 953, 636, 639],
    "Student Country": ["Ireland", "Portugal", "Portugal", "Italy", "Portugal", "Portugal", "Portugal", "Portugal", "Portugal", "Portugal", "Lithuania", "Italy", "Italy"],
    "Question ID": [77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77],
    "Type of Answer": [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    "Question Level": ["Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic", "Basic"],
    "Topic": ["Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics"],
    "Subtopic": ["Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics", "Statistics"],
    "Keywords": ["Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency", "Stem and Leaf diagram	Relative frequency	Sample	Frequency"]
}

# Load data into a pandas DataFrame
df = pd.DataFrame(data)

# Preprocess data (handle missing values - assuming no missing values here)

# Normalize scores (assuming scores are between 0 and 100)
df["Type of Answer"] = df["Type of Answer"] / 100  # Normalize scores (assuming correct answers are marked as 1)

# Feature extraction for knowledge points
def extract_features(data, knowledge_point):
  # Filter data for this knowledge point
  knowledge_point_data = data[data["Topic"] == knowledge_point]

  # Calculate features
  average_score = knowledge_point_data["Type of Answer"].mean()
  repetitions = knowledge_point_data.shape[0]  # Count rows for repetitions
  # Comments are not included in this sample data

  return {"average_score": average_score, "repetitions": repetitions}

# Knowledge Difficulty Model features dictionary
knowledge_difficulty_features = {}
for knowledge_point in df["Topic"].unique():
  features = extract_features(df, knowledge_point)
  knowledge_difficulty_features[knowledge_point] = features

# Now you have a dictionary 'knowledge_difficulty_features' with features for each knowledge point

# Further analysis and model building can be done using the extracted features
print(knowledge_difficulty_features)
