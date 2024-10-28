🚀 **Rocket Launch Success Predictor**

📑 **Project Overview**
This project aims to build a Rocket Launch Success Predictor using machine learning techniques. The model is trained on a dataset of past space mission launches to predict the likelihood of success for future rocket launches. The goal is to provide a tool that can assist organizations in making informed decisions about the risks and probabilities associated with launching rockets.

📊 **Dataset**
The dataset used for this project contains historical records of rocket launches, including information about:

Organization: The entity responsible for the launch (e.g., SpaceX, NASA).
Location: The site where the launch took place.
Date: The date and time of the launch.
Rocket Status: Operational status of the rocket (e.g., active, retired).
Price: Cost of the mission.
Mission Status: Outcome of the mission (e.g., success, failure, partial failure).

🔍 **Exploratory Data Analysis (EDA)**
To understand the data better and prepare it for modeling, several EDA steps were conducted:

Data Cleaning: Removed unnecessary columns, handled missing values, and converted relevant columns to numeric formats.
Data Visualization:
Distribution of Mission Prices: Analyzed the distribution of mission costs.
Correlation Heatmap: Explored relationships between different features.
Price vs. Mission Success: Examined the relationship between mission cost and outcome.
Success Rate by Organization: Analyzed which organizations had the highest and lowest success rates.
Impact of Launch Date and Rocket Status: Investigated if launch dates or rocket statuses influenced mission outcomes.

🧠 **Model Building**
A Logistic Regression Model was developed to predict the success rate of future rocket launches based on historical data. The steps included:

Data Preparation:
One-hot encoding of categorical variables.
Feature engineering (e.g., converting Mission Status to a binary numeric format).
Handling class imbalance and missing values.

Model Training:
The model was trained using the cleaned dataset to predict the probability of a successful launch.

Model Evaluation:
Evaluated the model's accuracy and fine-tuned it for better performance.

🔮 **How to Use the Model**
The trained model can predict the success rate of future rocket launches based on new input data (e.g., location, date, rocket status, price).

Input Data: Provide details such as the launch location, date, rocket status, and mission cost.
Output: The model returns a success rate percentage, which can be used for decision-making purposes (e.g., go/no-go decisions, risk assessments, and contingency planning).

📁 **Repository Structure**
data/: Contains the original and cleaned datasets used in the project.
notebooks/: Jupyter notebooks containing all EDA, data preprocessing, and model-building steps.
scripts/: Python scripts for training the model and running predictions.
model/: The saved trained model (.pkl file) for future use.
README.md: Project overview and documentation.

🚀 **Getting Started**
Clone the Repository:

Use the Model: Load the trained model from the model/ directory and use the provided scripts to input new data and get success predictions.

📈 **Results**
The model achieved an accuracy of around 90.6% on the test set, demonstrating its effectiveness in predicting the likelihood of a successful rocket launch. The model's performance can be further improved with additional data and feature engineering.

🤝 **Contributing**
Feel free to submit issues or pull requests for enhancements, bug fixes, or additional features. All contributions are welcome!

✉️ Contact
For questions or suggestions, please reach out to mminu0814@gmail.com

