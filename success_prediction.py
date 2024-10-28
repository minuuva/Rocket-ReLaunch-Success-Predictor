# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
file_path = '/Users/cmw/Downloads/mission_launches.csv'
data = pd.read_csv(file_path)
data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
data.head()

# %%
# Convert 'Mission_Status' to a binary classification: 1 for 'Success', 0 for others
data['Mission_Success'] = data['Mission_Status'].apply(lambda x: 1 if x == 'Success' else 0)

# Perform one-hot encoding on the categorical columns: 'Organisation', 'Location', and 'Rocket_Status'
encoded_data = pd.get_dummies(data, columns=['Organisation', 'Location', 'Rocket_Status'], drop_first=True)


# %%
# Step 2: Data Splitting
# Define the feature set (X) and target variable (y)
X = encoded_data.drop(columns=['Date', 'Detail', 'Price', 'Mission_Status', 'Mission_Success'])
y = encoded_data['Mission_Success']

# Split the dataset into training and testing sets with an 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Step 3: Model Building
# Initialize the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on the training set
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logistic_model.predict(X_test)

# %%
# Step 4: Model Evaluation
# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", confusion_mat)

# %%
import joblib

# Save the trained model
joblib.dump(logistic_model, 'launch_success_model.pkl')

# %%
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('launch_success_model.pkl')

# Function to preprocess input data and make predictions
def predict_launch_success():
    # Step 1: Collect Input Data from User
    location = input("Enter launch location (e.g., Kennedy_Space_Center): ")
    date = input("Enter launch date (e.g., Fri Aug 07, 2020 05:12 UTC): ")
    detail = input("Enter rocket detail (e.g., Falcon 9 Block 5 | Starlink V1 L9 & BlackSky): ")
    rocket_status = input("Enter rocket status (e.g., StatusActive): ")
    price = float(input("Enter price of the mission (e.g., 50.0): "))

    # Step 2: Prepare the Input Data for the Model
    input_data = {
        'Location_' + location: [1],   # Example: "Location_Kennedy_Space_Center"
        'Rocket_Status_' + rocket_status: [1],  # Example: "Rocket_Status_StatusActive"
        # Include other one-hot encoded columns based on the input...
        'Price': [price]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Add missing columns with 0 values
    all_columns = model.feature_names_in_
    for col in all_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure columns are in the same order as the model's expected input
    input_df = input_df[all_columns]

    # Step 3: Make Prediction Using the Loaded Model
    predicted_probability = model.predict_proba(input_df)[0][1]  # Probability of success

    # Return the success rate percentage
    print(f"Predicted Success Rate: {predicted_probability * 100:.2f}%")

# Run the prediction function
predict_launch_success()


