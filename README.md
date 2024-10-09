# Cattle Feedlot Machine Learning Analysis

## Dataset Overview

This project focuses on a simulated cattle feedlot dataset. The dataset includes information on individual cattle, such as their weight, age, feed type, and their readiness for market. The goal is to use machine learning to predict whether a particular cattle is ready for market based on various factors.

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

The goal of this project is to build a predictive machine learning model to classify whether cattle are "Market Ready" (1) or not (0). By analyzing various features such as weight, feed type, age, and daily feed intake, we aim to help farmers optimize their feeding process and determine the market readiness of cattle more accurately.

## Machine Learning Process

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

## Insights and Outcomes

After running the machine learning model, the following insights were gathered:

- **Model Performance**: The Random Forest Classifier achieved an accuracy of approximately 62.5%, with a moderate balance between precision and recall. However, the model struggled with identifying "Market Ready" cattle (class 1), which may suggest class imbalance in the dataset.
- **Feature Importance**: The features that had the most significant impact on predicting whether cattle were market-ready included:
  - **Days on Feed**: Cattle with longer feeding periods were more likely to be market-ready.
  - **Weight**: Heavier cattle showed a higher likelihood of being classified as ready for market.
  - **Average Gain Per Day**: Cattle with higher average weight gain also tended to be market-ready.
- **Improvements**: The model's performance could be further improved by addressing class imbalance, conducting more advanced feature engineering, and potentially gathering more data on underrepresented classes (market-ready cattle).

## Conclusion

This project demonstrates how machine learning can be applied to the cattle industry to predict market readiness based on various feeding and growth factors. By optimizing this process, farmers can make more informed decisions about when cattle are ready for sale, improving overall efficiency and profitability.

## Running the Code

To run this project, you will need:
1. The dataset (`cattle_feedlot_data.csv`) in the same directory as the Python script.
2. The Python script for machine learning analysis (`cattle_feedlot_ml_analysis.py`).

You can execute the script as follows:

```bash
python cattle_feedlot_ml_analysis.py
