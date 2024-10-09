# Cattle Feedlot Machine Learning Analysis

## Dataset Overview

This project focuses on a simulated cattle feedlot dataset that I created for my own practice. The dataset includes information on individual cattle, such as their weight, age, feed type, and their readiness for market. Additionally, we analyze the optimal amount of feed required for cattle based on various factors. The goal is to use machine learning to predict whether a particular cattle is ready for market and to predict the optimal daily feed amount.

### Features in the Dataset:
- **Cattle_ID**: Unique identifier for each cattle.
- **Weight**: Weight of the cattle in pounds.
- **Age_in_Months**: Age of the cattle in months.
- **Feed_Type**: The type of feed provided to the cattle (Grain, Grass, or Mixed).
- **Daily_Feed_Amount**: Amount of feed consumed daily in pounds.
- **Avg_Gain_Per_Day**: Average daily weight gain in pounds.
- **Days_on_Feed**: Number of days the cattle have been on feed.
- **Market_Ready**: A binary outcome (1 if market-ready, 0 if not).

## Goal

This project aims to achieve two primary objectives:
1. Build a predictive machine learning model to classify whether cattle are "Market Ready" (1) or not (0). By analyzing various features such as weight, feed type, age, and daily feed intake, we aim to help farmers optimize their feeding process and determine the market readiness of cattle more accurately.
2. Predict the **optimal daily feed amount** required by each cattle to maximize growth while maintaining efficiency, using regression modeling.

## Machine Learning Process

### Market Readiness Prediction:
1. **Data Preprocessing**:
   - Categorical variables (such as `Feed_Type`) were encoded using Label Encoding.
   - The dataset was split into training and testing sets to evaluate model performance.

2. **Model Selection**:
   - A **Random Forest Classifier** was selected as the baseline model due to its ability to handle non-linear relationships and feature interactions.

3. **Hyperparameter Tuning**:
   - We performed **RandomizedSearchCV** to find the best set of hyperparameters for the Random Forest model.
   - The following hyperparameters were tuned:
     - Number of trees (`n_estimators`)
     - Maximum depth (`max_depth`)
     - Minimum samples for a split (`min_samples_split`)
     - Minimum samples per leaf (`min_samples_leaf`)
     - Number of features considered for splits (`max_features`)

4. **Model Evaluation**:
   - The model was evaluated using **accuracy**, **precision**, **recall**, and **f1-score**. A **confusion matrix** was also used to assess the model’s classification performance.

### Feed Amount Prediction:
1. **Regression Model**:
   - A **Random Forest Regressor** was used to predict the **optimal daily feed amount**. The model used factors such as weight, age, average daily weight gain, and days on feed to predict how much feed should be provided to maximize growth.

2. **Model Evaluation**:
   - The performance of the regression model was evaluated using **Mean Squared Error (MSE)** and **R² score**. These metrics help assess how well the model fits the data and predicts the optimal feed amount.

## Insights and Outcomes

### Market Readiness Prediction:
- **Model Performance**: The Random Forest Classifier achieved an accuracy of approximately 62.5%, with a moderate balance between precision and recall. However, the model struggled with identifying "Market Ready" cattle (class 1), which may suggest class imbalance in the dataset.
- **Feature Importance**: The most significant factors affecting market readiness include:
  - **Days on Feed**: Longer feeding periods increased the likelihood of market readiness.
  - **Weight**: Heavier cattle were more likely to be market-ready.
  - **Average Gain Per Day**: Higher daily weight gain was associated with market readiness.

### Feed Amount Prediction:
- **Model Performance**: The Random Forest Regressor for predicting the optimal feed amount had a **Mean Squared Error (MSE)** of 114.79 and an **R² score** of -0.144, indicating that the model did not perform well in predicting feed amounts. This negative R² value suggests that the model was worse than a baseline mean prediction.
  
- **Why This Matters**: While the initial feed amount prediction model did not yield highly accurate results, exploring the optimal daily feed amount is crucial for improving efficiency in cattle feeding operations. Farmers can fine-tune the feeding process to maximize weight gain while avoiding overfeeding, reducing costs, and improving overall farm profitability.

- **Next Steps**: Improving the model's performance could involve gathering more data, exploring additional features, or trying different regression techniques (e.g., Gradient Boosting). Fine-tuning the model can help optimize feed efficiency and ensure the well-being of the cattle.

## Conclusion

This project demonstrates how machine learning can be applied to the cattle industry to predict both market readiness and the optimal feed amount. By optimizing these processes, farmers can make more informed decisions about when cattle are ready for sale and how much feed to provide, ultimately improving operational efficiency and profitability.
