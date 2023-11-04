# Import important stuff
import csv
import pandas as pd
import numpy as np 
from scipy.spatial import distance 
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay

# Select the relevant columns
df_combined = data[["distance_to_target_goal", "angle_to_target_goal", "shot_or_goal", "empty_net"]]

# Filter the test data for the 2019-2020 season
test_data = data[(data['about.dateTime'] >= '2019-10-01') & (data['about.dateTime'] < '2020-10-01')]

# Extract the corresponding columns for the test data
test_data = pd.concat([test_data, df_combined.loc[test_data.index]], axis=1)

# Filter the training and validation data for other seasons
train_val_data = data[~data.index.isin(test_data.index)]

# Extract the corresponding columns for the training and validation data
train_val_data = pd.concat([train_val_data, df_combined.loc[train_val_data.index]], axis=1)

# Split the remaining data into training and validation sets
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# Extract the corresponding columns for the training and validation sets
train_data = pd.concat([train_data, df_combined.loc[train_data.index]], axis=1)
val_data = pd.concat([val_data, df_combined.loc[val_data.index]], axis=1)


# Sélection de la caractéristique 'distance_to_target_goal' comme seule variable indépendante
train_data = train_data.dropna(subset=['distance_to_target_goal'])
train_data = train_data.loc[:, ~train_data.columns.duplicated()]
X = train_data[['distance_to_target_goal']]
y = train_data['shot_or_goal']

# Créer et entraîner un modèle de régression logistique avec les paramètres par défaut
clf = LogisticRegression()
clf.fit(X, y)

# Sélection de la caractéristique 'distance_to_target_goal' dans l'ensemble de validation
val_data = val_data.dropna(subset=['distance_to_target_goal'])
val_data = val_data.loc[:, ~val_data.columns.duplicated()]
X_val = val_data[['distance_to_target_goal']] 
y_val_true = val_data['shot_or_goal']

# Faire des prédictions sur l'ensemble de validation
y_val_pred = clf.predict(X_val)

# Calcul de l'accuracy
accuracy = accuracy_score(y_val_true, y_val_pred)

print("Accuracy sur l'ensemble de validation :", accuracy)

#FPR is false positive rate, TPR is true positive rate
fpr, tpr, thresholds = roc_curve(y_val_true, clf.predict_proba(X_val)[:, 1])
auc = roc_auc_score(y_val_true, clf.predict_proba(X_val)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Predict probabilities for the validation data
val_data['probability'] = clf.predict_proba(X_val)[:, 1]

# Calculate the goal rate as a function of percentiles on the validation set
percentiles = np.arange(0, 101, 10)[::-1]  
goal_rates = []

for percentile in percentiles:
    # Calculate the percentile threshold for the predicted probabilities
    threshold = np.percentile(val_data['probability'], percentile)

    # Create a binary column based on whether the probability is above the threshold
    val_data['predicted_goal'] = (val_data['probability'] > threshold).astype(int)

    # Calculate the goal rate for this percentile on the validation set
    goal_rate = val_data[val_data['predicted_goal'] == 1]['shot_or_goal'].mean()
    goal_rates.append(goal_rate * 100)  

# Create a DataFrame to store the results
result_df = pd.DataFrame({
    'Percentile': percentiles,
    'Goal Rate': goal_rates
})

# Create a graph for the validation set
plt.figure(figsize=(10, 6))
plt.plot(result_df['Percentile'], result_df['Goal Rate'], marker='o')
plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Goal Rate (%)')
plt.title('Goal Rate vs. Shot Probability Model Percentile')
plt.grid(True)
plt.gca().invert_xaxis() 
plt.ylim(0, 100)  # Set y-axis limits to 0% to 100%
plt.show()


# Calculate the cumulative proportion of goals as a function of percentiles
percentiles = np.arange(0, 101, 10)[::-1]  # Define percentiles in reverse order (100% to 0)
cumulative_goal_proportions = []

# Calculate the total number of goals in the dataset
total_goals = val_data['shot_or_goal'].sum()

for percentile in percentiles:
    # Predict probabilities for the validation data
    val_data['probability'] = clf.predict_proba(X_val)[:, 1]

    # Calculate the percentile threshold for the predicted probabilities
    threshold = np.percentile(val_data['probability'], percentile)

    # Create a binary column based on whether the probability is above the threshold
    val_data['predicted_goal'] = (val_data['probability'] > threshold).astype(int)

    # Calculate the cumulative proportion of goals for this percentile
    cumulative_proportion = val_data[val_data['predicted_goal'] == 1]['shot_or_goal'].sum() / total_goals
    cumulative_goal_proportions.append(cumulative_proportion * 100)  # Convert to percentage

# Create a DataFrame to store the results
result_df = pd.DataFrame({
    'Percentile': percentiles,
    'Cumulative Goal Proportion': cumulative_goal_proportions
})

# Create a graph for the validation set
plt.figure(figsize=(10, 6))
plt.plot(result_df['Percentile'], result_df['Cumulative Goal Proportion'], marker='o', label='Cumulative Goal Proportion')
plt.xlabel('Shot Probability Model Percentile (100% to 0%)')
plt.ylabel('Percentage (%)')
plt.title('Cumulative Goal Proportion vs. Shot Probability Model Percentile')
plt.grid(True)
plt.gca().invert_xaxis()  
plt.ylim(0, 100)  
plt.legend(loc='upper right')  # Add a legend
plt.show()

# Calculate predicted probabilities
y_val_prob = clf.predict_proba(X_val)[:, 1]

# Create a CalibrationDisplay using the .from_predictions() method
calibration_display = CalibrationDisplay.from_predictions(y_val_true, y_val_prob, n_bins=50, pos_label=None, name="Model 1 - Distance", ref_line=True)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the reliability diagram
calibration_display.plot(ax=ax, name="Model 1 - Distance", ref_line=True)

# Customize the plot and add labels
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Reliability Diagram (Calibration Curve) - Model 1")

# Show the plot
plt.show()


# Select the 'angle_to_target_goal' feature as the only independent variable in the training dataset
train_data = train_data.dropna(subset=['angle_to_target_goal'])
X = train_data[['angle_to_target_goal']]
y = train_data['shot_or_goal']

# Create and train a Logistic Regression model with default parameters
clf2 = LogisticRegression()
clf2.fit(X, y)

# Select the 'angle_to_target_goal' feature in the validation dataset
val_data = val_data.dropna(subset=['angle_to_target_goal'])
X_val2 = val_data[['angle_to_target_goal']]
y_val_true2 = val_data['shot_or_goal']

# Make predictions on the validation dataset
y_val_pred2 = clf2.predict(X_val2)

fpr2, tpr2, thresholds2 = roc_curve(y_val_true2, clf2.predict_proba(X_val2)[:, 1])
auc2 = roc_auc_score(y_val_true2, clf2.predict_proba(X_val2)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr2, tpr2, label=f'Logistic Regression (AUC = {auc2:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Predict probabilities for the validation data
val_data['probability2'] = clf2.predict_proba(X_val2)[:, 1]  

# Calculate the goal rate as a function of percentiles on the validation set
percentiles = np.arange(0, 101, 10)[::-1]  # Define percentiles in reverse order (100 to 0)
goal_rates = []

for percentile in percentiles:
    # Calculate the percentile threshold for the predicted probabilities
    threshold = np.percentile(val_data['probability2'], percentile)  # Use the new probability variable

    # Create a binary column based on whether the probability is above the threshold
    val_data['predicted_goal2'] = (val_data['probability2'] > threshold).astype(int)  

    # Calculate the goal rate for this percentile on the validation set
    goal_rate = val_data[val_data['predicted_goal2'] == 1]['shot_or_goal'].mean()  
    goal_rates.append(goal_rate * 100)  

# Create a DataFrame to store the results
result_df = pd.DataFrame({
    'Percentile': percentiles,
    'Goal Rate': goal_rates
})

# Create a graph for the validation set
plt.figure(figsize=(10, 6))
plt.plot(result_df['Percentile'], result_df['Goal Rate'], marker='o')
plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Goal Rate (%)')
plt.title('Goal Rate vs. Shot Probability Model Percentile')
plt.grid(True)
plt.gca().invert_xaxis() 
plt.ylim(0, 100)  
plt.show()


# Calculate the cumulative proportion of goals as a function of percentiles for clf2 (angle_to_target_goal)
percentiles = np.arange(0, 101, 10)[::-1]  # Define percentiles in reverse order (100% to 0)
cumulative_goal_proportions = []

# Calculate the total number of goals in the dataset
total_goals = val_data['shot_or_goal'].sum()

for percentile in percentiles:
    # Predict probabilities for the validation data using clf2
    val_data['probability2'] = clf2.predict_proba(X_val2)[:, 1]  

    # Calculate the percentile threshold for the predicted probabilities
    threshold = np.percentile(val_data['probability2'], percentile)  

    # Create a binary column based on whether the probability is above the threshold
    val_data['predicted_goal2'] = (val_data['probability2'] > threshold).astype(int)  

    # Calculate the cumulative proportion of goals for this percentile
    cumulative_proportion = val_data[val_data['predicted_goal2'] == 1]['shot_or_goal'].sum() / total_goals  
    cumulative_goal_proportions.append(cumulative_proportion * 100)  

# Create a DataFrame to store the results
result_df = pd.DataFrame({
    'Percentile': percentiles,
    'Cumulative Goal Proportion': cumulative_goal_proportions
})

# Create a graph for the validation set
plt.figure(figsize=(10, 6))
plt.plot(result_df['Percentile'], result_df['Cumulative Goal Proportion'], marker='o', label='Cumulative Goal Proportion')
plt.xlabel('Shot Probability Model Percentile (100% to 0%)')
plt.ylabel('Percentage (%)')
plt.title('Cumulative Goal Proportion vs. Shot Probability Model Percentile (Angle-to-Goal)')
plt.grid(True)
plt.gca().invert_xaxis()  
plt.ylim(0, 100)  
plt.legend(loc='upper right')  
plt.show()

# Combine 'distance_to_target_goal' and 'angle_to_target_goal' as independent variables
X_combined = train_data[['distance_to_target_goal', 'angle_to_target_goal']]

# Define the target variable
y_combined = train_data['shot_or_goal']

# Create and train the logistic regression model
clf3 = LogisticRegression()
clf3.fit(X_combined, y_combined)

# Prepare the validation data with the same features
X_val_combined = val_data[['distance_to_target_goal', 'angle_to_target_goal']]
y_val_true_combined = val_data['shot_or_goal']

# Make predictions on the validation set using the new model
y_val_pred_combined = clf3.predict(X_val_combined)

fpr3, tpr3, thresholds3 = roc_curve(y_val_true_combined, clf3.predict_proba(X_val_combined)[:, 1])
auc3 = roc_auc_score(y_val_true_combined, clf3.predict_proba(X_val_combined)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr3, tpr3, label=f'Logistic Regression (AUC = {auc3:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Predict probabilities for the validation data using clf3
val_data['probability3'] = clf3.predict_proba(X_val_combined)[:, 1]  

# Calculate the goal rate as a function of percentiles on the validation set
percentiles = np.arange(0, 101, 10)[::-1]  # Define percentiles in reverse order (100 to 0)
goal_rates = []

for percentile in percentiles:
    # Calculate the percentile threshold for the predicted probabilities
    threshold = np.percentile(val_data['probability3'], percentile)  # Use the new probability variable

    # Create a binary column based on whether the probability is above the threshold
    val_data['predicted_goal3'] = (val_data['probability3'] > threshold).astype(int)  

    # Calculate the goal rate for this percentile on the validation set
    goal_rate = val_data[val_data['predicted_goal3'] == 1]['shot_or_goal'].mean() 
    goal_rates.append(goal_rate * 100)  

# Create a DataFrame to store the results
result_df3 = pd.DataFrame({
    'Percentile': percentiles,
    'Goal Rate': goal_rates
})

# Create a graph for the validation set
plt.figure(figsize=(10, 6))
plt.plot(result_df3['Percentile'], result_df3['Goal Rate'], marker='o')
plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Goal Rate (%)')
plt.title('Goal Rate vs. Shot Probability Model Percentile')
plt.grid(True)
plt.gca().invert_xaxis() 
plt.ylim(0, 100)  
plt.show()


# Calculate the cumulative proportion of goals as a function of percentiles for clf3 (distance_to_target_goal and angle_to_target_goal)
percentiles = np.arange(0, 101, 10)[::-1]  
cumulative_goal_proportions = []

# Calculate the total number of goals in the dataset
total_goals = val_data['shot_or_goal'].sum()

for percentile in percentiles:
    # Predict probabilities for the validation data using clf3
    val_data['probability3'] = clf3.predict_proba(X_val_combined)[:, 1]  

    # Calculate the percentile threshold for the predicted probabilities
    threshold = np.percentile(val_data['probability3'], percentile)  

    # Create a binary column based on whether the probability is above the threshold
    val_data['predicted_goal3'] = (val_data['probability3'] > threshold).astype(int)  

    # Calculate the cumulative proportion of goals for this percentile
    cumulative_proportion = val_data[val_data['predicted_goal3'] == 1]['shot_or_goal'].sum() / total_goals  
    cumulative_goal_proportions.append(cumulative_proportion * 100)  

# Create a DataFrame to store the results
result_df3 = pd.DataFrame({
    'Percentile': percentiles,
    'Cumulative Goal Proportion': cumulative_goal_proportions
})

# Create a graph for the validation set
plt.figure(figsize=(10, 6))
plt.plot(result_df3['Percentile'], result_df3['Cumulative Goal Proportion'], marker='o', label='Cumulative Goal Proportion')
plt.xlabel('Shot Probability Model Percentile (100% to 0%)')
plt.ylabel('Percentage (%)')
plt.title('Cumulative Goal Proportion vs. Shot Probability Model Percentile (Distance-to-Goal and Angle-to-Goal)')
plt.grid(True)
plt.gca().invert_xaxis()  
plt.ylim(0, 100)  
plt.legend(loc='upper right')  
plt.show()

# Generate random predicted probabilities for the random baseline
random_probabilities = np.random.uniform(0, 1, len(X_val2))

# Calculate ROC curve and AUC for the random baseline
fpr_random, tpr_random, thresholds_random = roc_curve(y_val_true, random_probabilities)
roc_auc_random = roc_auc_score(y_val_true, random_probabilities)

# Create the ROC curve plot for the random baseline
plt.figure(figsize=(8, 6))
plt.plot(fpr_random, tpr_random, label=f'Random Baseline (AUC = {roc_auc_random:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Random Baseline)')
plt.legend()
plt.show()

# Calculate the goal rate as a function of percentiles for the descending threshold-based baseline
thresholds_baseline = np.percentile(random_probabilities, percentiles)
goal_rates_baseline = []

for threshold in thresholds_baseline:
    # Create a binary column based on whether the probability is above the threshold
    predicted_goal_baseline = (random_probabilities > threshold).astype(int)

    # Calculate the goal rate for this threshold
    goal_rate_baseline = np.mean(predicted_goal_baseline)
    goal_rates_baseline.append(goal_rate_baseline * 100)  # Convert to percentage

# Create a DataFrame to store the results for the descending threshold-based baseline
result_df_baseline = pd.DataFrame({
    'Percentile': percentiles,
    'Goal Rate (Descending Threshold-Based Baseline)': goal_rates_baseline
})

# Create a graph for the descending threshold-based baseline
plt.figure(figsize=(10, 6))
plt.plot(result_df_baseline['Percentile'], result_df_baseline['Goal Rate (Descending Threshold-Based Baseline)'], marker='o')
plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Goal Rate (%)')
plt.title('Goal Rate vs. Shot Probability Model Percentile (Descending Threshold-Based Baseline)')
plt.grid(True)
plt.gca().invert_xaxis() 
plt.ylim(0, 100)  
plt.show()

# Calculate the cumulative proportion of goals as a function of percentiles for the random baseline
cumulative_goal_proportions_random = []

total_goals_random = np.sum(predicted_goal_random)  # Calculate the total number of goals

for percentile in percentiles:
    # Calculate the percentile threshold for the random probabilities
    threshold = np.percentile(random_probabilities, percentile)

    # Create a binary column based on whether the probability is above the threshold
    predicted_goal_random = (random_probabilities > threshold).astype(int)

    # Calculate the cumulative proportion of goals for this percentile
    cumulative_proportion_random = np.sum(predicted_goal_random) / total_goals_random
    cumulative_goal_proportions_random.append(cumulative_proportion_random * 100)  

# Create a DataFrame to store the results for the random baseline
result_df_cumulative_random = pd.DataFrame({
    'Percentile': percentiles,
    'Cumulative Goal Proportion (Random Baseline)': cumulative_goal_proportions_random
})

# Create a graph for the cumulative goal proportions for the random baseline
plt.figure(figsize=(10, 6))
plt.plot(result_df_cumulative_random['Percentile'], result_df_cumulative_random['Cumulative Goal Proportion (Random Baseline)'], marker='o', label='Cumulative Goal Proportion (Random Baseline)')
plt.xlabel('Shot Probability Model Percentile (100% to 0%)')
plt.ylabel('Percentage (%)')
plt.title('Cumulative Goal Proportion vs. Shot Probability Model Percentile (Random Baseline)')
plt.grid(True)
plt.gca().invert_xaxis()  
plt.ylim(0, 100)  
plt.legend(loc='upper right')  
plt.show()


# Calculate ROC curve and AUC for clf
fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
roc_auc = roc_auc_score(y_val_true, y_val_prob)

# Calculate ROC curve and AUC for clf2
fpr2, tpr2, thresholds2 = roc_curve(y_val_true2, clf2.predict_proba(X_val2)[:, 1])
roc_auc2 = roc_auc_score(y_val_true2, clf2.predict_proba(X_val2)[:, 1])

# Calculate ROC curve and AUC for clf3
fpr3, tpr3, thresholds3 = roc_curve(y_val_true_combined, clf3.predict_proba(X_val_combined)[:, 1])
roc_auc3 = roc_auc_score(y_val_true_combined, clf3.predict_proba(X_val_combined)[:, 1])

# Calculate ROC curve and AUC for random baseline
fpr_random, tpr_random, thresholds_random = roc_curve(y_val_true, random_probabilities)
roc_auc_random = roc_auc_score(y_val_true, random_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model 1 - Distance (AUC = {roc_auc:.2f})')
plt.plot(fpr2, tpr2, label=f'Model 2 - Angle (AUC = {roc_auc2:.2f})')
plt.plot(fpr3, tpr3, label=f'Model 3 - Distance and Angle (AUC = {roc_auc3:.2f})')
plt.plot(fpr_random, tpr_random, label=f'Random Classifier (AUC = {roc_auc_random:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()

# Correct the percentile values
percentiles = np.arange(0, 101, 10)[::-1]  # Define percentiles in reverse order (100 to 0)

# Calculate the goal rates for clf
goal_rates = []
for percentile in percentiles:
    threshold = np.percentile(val_data['probability'], percentile)
    val_data['predicted_goal'] = (val_data['probability'] > threshold).astype(int)
    goal_rate = val_data[val_data['predicted_goal'] == 1]['shot_or_goal'].mean()
    goal_rates.append(goal_rate * 100)

# Calculate the goal rates for clf2
goal_rates2 = []
for percentile in percentiles:
    threshold = np.percentile(val_data['probability2'], percentile)
    val_data['predicted_goal2'] = (val_data['probability2'] > threshold).astype(int)
    goal_rate2 = val_data[val_data['predicted_goal2'] == 1]['shot_or_goal'].mean()
    goal_rates2.append(goal_rate2 * 100)

# Calculate the goal rates for clf3
goal_rates3 = []
for percentile in percentiles:
    threshold = np.percentile(val_data['probability3'], percentile)
    val_data['predicted_goal3'] = (val_data['probability3'] > threshold).astype(int)
    goal_rate3 = val_data[val_data['predicted_goal3'] == 1]['shot_or_goal'].mean()
    goal_rates3.append(goal_rate3 * 100)

# Calculate the goal rates for the random baseline
thresholds_baseline = np.percentile(random_probabilities, percentiles)
goal_rates_random = []
for threshold in thresholds_baseline:
    predicted_goal_baseline = (random_probabilities > threshold).astype(int)
    goal_rate_baseline = np.mean(predicted_goal_baseline)
    goal_rates_random.append(goal_rate_baseline * 100)

# Create a DataFrame to store the results for all goal rates
result_df_combined = pd.DataFrame({
    'Percentile': percentiles,
    'Goal Rate (clf)': goal_rates,
    'Goal Rate (clf2)': goal_rates2,
    'Goal Rate (clf3)': goal_rates3,
    'Goal Rate (Random Baseline)': goal_rates_random
})

# Create a graph to display all the goal rates
plt.figure(figsize=(10, 6))
plt.plot(result_df_combined['Percentile'], result_df_combined['Goal Rate (clf)'], marker='o', label='Model 1 - Distance')
plt.plot(result_df_combined['Percentile'], result_df_combined['Goal Rate (clf2)'], marker='o', label='Model 2 - Angle')
plt.plot(result_df_combined['Percentile'], result_df_combined['Goal Rate (clf3)'], marker='o', label='Model 3 - Distance and Angle')
plt.plot(result_df_combined['Percentile'], result_df_combined['Goal Rate (Random Baseline)'], marker='o', label='Random Baseline')
plt.xlabel('Shot Probability Model Percentile')
plt.ylabel('Goal Rate (%)')
plt.title('Goal Rate vs. Shot Probability Model Percentile')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.ylim(0, 100)  # Set y-axis limits to 0% to 100%
plt.show()

# Calculate the cumulative proportion of goals as a function of percentiles for clf
cumulative_goal_proportions = []
total_goals = val_data['shot_or_goal'].sum()

for percentile in percentiles:
    threshold = np.percentile(val_data['probability'], percentile)
    predicted_goal = (val_data['probability'] > threshold).astype(int)
    cumulative_proportion = val_data[predicted_goal == 1]['shot_or_goal'].sum() / total_goals
    cumulative_goal_proportions.append(cumulative_proportion * 100)

# Calculate the cumulative proportion of goals for clf2
cumulative_goal_proportions2 = []
total_goals2 = val_data['shot_or_goal'].sum()

for percentile in percentiles:
    threshold = np.percentile(val_data['probability2'], percentile)
    predicted_goal2 = (val_data['probability2'] > threshold).astype(int)
    cumulative_proportion2 = val_data[predicted_goal2 == 1]['shot_or_goal'].sum() / total_goals2
    cumulative_goal_proportions2.append(cumulative_proportion2 * 100)

# Calculate the cumulative proportion of goals for clf3
cumulative_goal_proportions3 = []
total_goals3 = val_data['shot_or_goal'].sum()

for percentile in percentiles:
    threshold = np.percentile(val_data['probability3'], percentile)
    predicted_goal3 = (val_data['probability3'] > threshold).astype(int)
    cumulative_proportion3 = val_data[predicted_goal3 == 1]['shot_or_goal'].sum() / total_goals3
    cumulative_goal_proportions3.append(cumulative_proportion3 * 100)

# Calculate the cumulative proportion of goals for the random baseline
cumulative_goal_proportions_random = []
total_goals_random = np.sum(predicted_goal_random)

for percentile in percentiles:
    threshold = np.percentile(random_probabilities, percentile)
    predicted_goal_random = (random_probabilities > threshold).astype(int)
    cumulative_proportion_random = np.sum(predicted_goal_random) / total_goals_random
    cumulative_goal_proportions_random.append(cumulative_proportion_random * 100)

# Create a DataFrame to store the results for all cumulative goal proportions
result_df_combined_cumulative = pd.DataFrame({
    'Percentile': percentiles,
    'Cumulative Goal Proportion (clf)': cumulative_goal_proportions,
    'Cumulative Goal Proportion (clf2)': cumulative_goal_proportions2,
    'Cumulative Goal Proportion (clf3)': cumulative_goal_proportions3,
    'Cumulative Goal Proportion (Random Baseline)': cumulative_goal_proportions_random
})

# Create a graph to display all the cumulative goal proportions
plt.figure(figsize=(10, 6))
plt.plot(result_df_combined_cumulative['Percentile'], result_df_combined_cumulative['Cumulative Goal Proportion (clf)'], marker='o', label='Model 1 - Distance')
plt.plot(result_df_combined_cumulative['Percentile'], result_df_combined_cumulative['Cumulative Goal Proportion (clf2)'], marker='o', label='Model 2 - Angle')
plt.plot(result_df_combined_cumulative['Percentile'], result_df_combined_cumulative['Cumulative Goal Proportion (clf3)'], marker='o', label='Model 3 - Distance and Angle')
plt.plot(result_df_combined_cumulative['Percentile'], result_df_combined_cumulative['Cumulative Goal Proportion (Random Baseline)'], marker='o', label='Random Baseline')

plt.xlabel('Shot Probability Model Percentile (100% to 0%)')
plt.ylabel('Percentage (%)')
plt.title('Cumulative Goal Proportion vs. Shot Probability Model Percentile')
plt.grid(True)
plt.gca().invert_xaxis()
plt.ylim(0, 100)  # Set y-axis limits to range from 0% to 100%
plt.legend(loc='upper left') 
plt.show()

# Calculate predicted probabilities
y_val_prob = clf.predict_proba(X_val)[:, 1]
y_val_prob2 = clf2.predict_proba(X_val2)[:, 1] 
y_val_prob3 = clf3.predict_proba(X_val_combined)[:, 1]  

# Generate random predicted probabilities for the random baseline
random_probabilities = np.random.uniform(0, 1, len(X_val))

# Create a CalibrationDisplay using the .from_predictions() method for each model
calibration_display = CalibrationDisplay.from_predictions(y_val_true, y_val_prob, n_bins=50, pos_label=None, name="Model 1 - Distance", ref_line=True)
calibration_display2 = CalibrationDisplay.from_predictions(y_val_true2, y_val_prob2, n_bins=50, pos_label=None, name="Model 2", ref_line=True)
calibration_display3 = CalibrationDisplay.from_predictions(y_val_true_combined, y_val_prob3, n_bins=50, pos_label=None, name="Model 3", ref_line=True)
calibration_display_random = CalibrationDisplay.from_predictions(y_val_true, random_probabilities, n_bins=50, pos_label=None, name="Random Baseline", ref_line=True)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the reliability diagrams for each model
calibration_display.plot(ax=ax, name="Model 1 - Distance", ref_line=True)
calibration_display2.plot(ax=ax, name="Model 2 - Angle", ref_line=True)
calibration_display3.plot(ax=ax, name="Model 3 - Distance and Angle", ref_line=True)
calibration_display_random.plot(ax=ax, name="Random Baseline", ref_line=True)

# Customize the plot and add labels
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Reliability Diagram (Calibration Curve) for Multiple Models")

# Show the plot
plt.legend()
plt.show()
