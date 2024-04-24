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
