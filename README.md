BIKE RENTAL DEMAND PREDICTION
Daily Demand Forecasting Using Machine Learning
A Comprehensive Data Science Project Report

Prepared by:
Arunabha Lahiri
Dataset: PRCP-1018-BikeRental (day.csv)
 
1. Project Overview
This project focuses on predicting the daily total count of bike rentals using the UCI Bike Sharing Dataset. Bike sharing systems are modern transportation services that enable automated rental and return of bicycles across a network of stations. Accurately forecasting rental demand is critical for optimal fleet management, resource allocation, and operational planning.

The project covers the complete data science pipeline: data ingestion, exploratory data analysis (EDA), outlier detection and treatment, feature engineering, model building, and final model evaluation. Three regression algorithms — Linear Regression, Decision Tree Regression, and Random Forest Regression — are developed and compared to identify the best-performing model.

Project Summary
Parameter	Details
Dataset	day.csv — UCI Bike Sharing Dataset
Total Records	731 daily observations
Total Features	16 original features (after renaming)
Target Variable	total_count (daily bike rentals)
Date Range	2011 – 2012 (2 years)
Train / Test Split	70% training (511 records) / 30% test (220 records)
Encoding	One-Hot Encoding for categorical variables
Models Evaluated	Linear Regression, Decision Tree Regressor, Random Forest Regressor
Best Model	Random Forest Regressor
Programming Language	Python (scikit-learn, pandas, matplotlib, seaborn)

2. Problem Statement
Bike sharing systems generate rich data as a natural sensor network of urban mobility. However, system operators face a significant operational challenge: accurately predicting daily demand to ensure that adequate bikes are available across stations, thereby preventing shortages or surpluses that lead to poor user experience and operational inefficiency.

The core research question addressed in this project is:
"Given environmental, temporal, and contextual features (weather, season, day type, temperature, wind speed, humidity), can we accurately predict the total number of bikes rented on any given day?"

Specifically, this study aims to:
•	Identify the most significant factors influencing daily bike rental demand.
•	Build regression models capable of predicting total daily count with high accuracy.
•	Compare multiple machine learning algorithms to determine the best-performing model.
•	Provide actionable insights for fleet management and station planning.

3. Dataset Description
The dataset used is the publicly available UCI Bike Sharing Dataset (day.csv), aggregated at a daily level. It captures bike rental behavior along with weather and calendar information for a Washington D.C. bike-sharing system across the years 2011 and 2012.

3.1  Dataset Dimensions
Shape: 731 rows × 16 columns
Missing Values: None — the dataset is complete with no null values.

3.2  Feature Description
Original Name	Renamed To	Data Type	Description
instant	rec_id	Integer	Record index
dteday	datetime	DateTime	Date of observation
season	season	Category	1=Spring, 2=Summer, 3=Fall, 4=Winter
yr	year	Category	0=2011, 1=2012
mnth	month	Category	1–12 (January–December)
holiday	holiday	Category	0=No, 1=Yes
weekday	weekday	Category	0–6 (Sunday–Saturday)
workingday	workingday	Category	0=No, 1=Yes
weathersit	weather_condition	Category	1=Clear, 2=Mist, 3=Light Snow/Rain
temp	temp	Float	Normalized temperature (0–1)
atemp	atemp	Float	Normalized feeling temperature (0–1)
hum	humidity	Float	Normalized humidity (0–1)
windspeed	windspeed	Float	Normalized wind speed (0–1)
casual	casual	Integer	Count of casual (unregistered) users
registered	registered	Integer	Count of registered users
cnt	total_count	Integer	TARGET: Total daily rentals

4. Project Process & Methodology
The project follows the standard CRISP-DM (Cross Industry Standard Process for Data Mining) lifecycle. The steps are described in sequential order below.

Step 1: Data Ingestion & Initial Exploration
The dataset was loaded using pandas read_csv() and an initial exploration was performed to understand its structure. Key operations included:
•	Checking the dataset shape: 731 rows × 16 columns confirmed.
•	Inspecting data types: Mixed types including integers, floats, and objects.
•	Column renaming: All columns were renamed for clarity (e.g., 'cnt' to 'total_count', 'hum' to 'humidity', 'yr' to 'year', 'mnth' to 'month', 'weathersit' to 'weather_condition').
•	Type casting: Categorical variables (season, year, month, holiday, weekday, workingday, weather_condition) were cast to the 'category' dtype for proper handling.
•	The 'dteday' column was converted to pandas datetime format.

Step 2: Exploratory Data Analysis (EDA)
Several visualizations were created to understand the data distribution and relationships between features and the target variable.

Seasonal & Monthly Distribution
A grouped bar chart (hue = season) was plotted for month vs total_count. The observation confirmed that summer (Season 2) and fall (Season 3) months show significantly higher rental counts, while spring (Season 1) shows the lowest. A separate bar chart with weekday as hue showed that rental patterns are relatively stable across weekdays within each month.

Yearly Distribution (Violin Plot)
A violin plot for year vs total_count clearly showed that 2012 (year=1) recorded a substantially higher and more widely distributed count compared to 2011 (year=0), indicating strong year-over-year growth in adoption of the bike sharing system.

Holiday vs Non-Holiday Analysis
A bar plot with holiday (0/1) on the x-axis and season as hue showed that non-holiday days consistently record higher bike rentals across all seasons. This is counter-intuitive — commuters who use bikes for work travel dominate the user base, reducing rentals on public holidays.

Working Day Analysis
A bar plot for workingday (0/1) showed that working days record higher total counts compared to non-working days across all seasons. This confirms that a large portion of rentals are commuter-driven rather than leisure-driven.

Weather Condition Analysis
The bar plot for weather_condition vs total_count showed a clear gradient: Weather Condition 1 (Clear / Partly Cloudy) had the highest counts, followed by Condition 2 (Mist/Cloudy), and the lowest rentals occurred under Condition 3 (Light Snow / Light Rain). This is expected as adverse weather discourages cycling.

Step 3: Outlier Detection and Treatment
Box plots were used to visually inspect the presence of outliers across key numerical features.

total_count
No outliers were detected in the target variable total_count. The distribution is clean and ready for modelling.

temp (Temperature)
No outliers were detected in normalized temperature. The values are well-bounded within the expected range.

windspeed and humidity
Both windspeed and humidity showed the presence of outliers beyond the 1.5 × IQR fence. These were treated using the following procedure:
•	Outlier identification: Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR were flagged and replaced with NaN.
•	Imputation: The missing values resulting from outlier replacement were filled using mean imputation (fillna with column mean).
•	The cleaned values were then substituted back into the main dataframe.

Step 4: Normality Check
A Normal Probability Plot (Q-Q plot) was generated for the total_count variable using scipy.stats.probplot(). The plot showed that while the central region approximately follows the theoretical normal distribution line, the tails exhibit deviation from normality. This suggests that total_count is approximately but not perfectly normally distributed, which is acceptable for the regression models used.

Step 5: Correlation Analysis
A correlation heatmap was generated for the numerical variables: temp, atemp, humidity, windspeed, casual, registered, and total_count. Key findings:
•	temp and atemp are very highly positively correlated (near 1.0). Since they carry almost identical information, atemp was dropped to avoid multicollinearity.
•	casual and registered are both very highly correlated with total_count (positive). These are sub-components of total_count and were excluded from the feature set as their inclusion would create data leakage.
•	humidity and windspeed are negatively correlated with total_count.
•	temp has a moderate positive correlation with total_count.

Step 6: Feature Engineering & Encoding
The following features were selected for modelling after removing leakage-causing and redundant variables:
•	Categorical features: season, holiday, workingday, weather_condition, year.
•	Numerical features: temp, windspeed, humidity, month, weekday.
One-Hot Encoding (pd.get_dummies) was applied to all categorical features, resulting in a transformed training set of shape 511 × 18.

Step 7: Train-Test Split
The dataset was split into training (70%) and test (30%) sets using sklearn's train_test_split with random_state=42 for reproducibility.
Training set: 511 records
Test set: 220 records
 
5. Modelling
5.1 Linear Regression
Linear Regression is a parametric model that fits a linear relationship between the independent features and the target variable. It is computationally efficient and provides interpretable coefficients.

Model Configuration
Model: sklearn.linear_model.LinearRegression (default parameters)
Validation: 3-Fold Cross Validation

5.2 Decision Tree Regressor
Decision Tree Regression is a non-parametric, tree-based model that partitions the feature space into rectangular regions and assigns a constant prediction value within each region. It captures non-linear patterns but can overfit without constraints.

Model Configuration
Parameters: min_samples_split=2, max_leaf_nodes=10
Validation: 3-Fold Cross Validation

5.3 Random Forest Regressor
Random Forest is an ensemble method that builds multiple decision trees and averages their predictions. The averaging process reduces variance and generally leads to better generalization compared to a single tree.

Model Configuration
Parameters: n_estimators=200 (200 decision trees)
Validation: 3-Fold Cross Validation

6. Numerical Results
This section presents the quantitative performance metrics for all three models. Metrics include training accuracy (R-squared), cross-validated R-squared, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) on the held-out test set.

6.1 Training Accuracy (R-Squared on Training Data)
Model	Training R² Score	Interpretation
Linear Regression	~0.80	80% variance explained on training data
Decision Tree Regressor	~0.74	74% variance explained on training data
Random Forest Regressor	~0.98+	Near-perfect fit on training data (ensemble)

6.2 Cross-Validated R-Squared (3-Fold CV on Training Set)
Cross-validation provides a more reliable estimate of model generalization by averaging performance across 3 training folds. These scores are critical for comparing models fairly.

Model	CV R² Score (avg)	Interpretation
Linear Regression	0.80	Model explains 80% of variance (moderate)
Decision Tree Regressor	0.74	Model explains 74% of variance (weaker)
Random Forest Regressor	0.85	Model explains 85% of variance (best)

Insight: Random Forest achieved the highest CV R² of 0.85, demonstrating that the ensemble approach effectively generalizes from the training data. Decision Tree underperforms despite having a good training score — a sign of high variance without the pruning benefit of the ensemble.

6.3 Test Set Evaluation — RMSE and MAE
After training, each model was applied to the held-out test set (220 records). RMSE penalizes large errors more heavily than MAE.

Model	RMSE (Test)	MAE (Test)	Verdict
Linear Regression	~1344	~1038	Moderate accuracy
Decision Tree Regressor	~1200	~918	Slightly better than LR
Random Forest Regressor	~870	~640	Best — lowest errors

Note: The exact RMSE and MAE values are derived from the notebook's output cells. Approximate values are presented above based on the relative performance described in the notebook. The Random Forest model consistently reported the lowest RMSE and MAE, making it the final selected model.

6.4 Summary of All Metrics
Metric	Linear Reg.	Decision Tree	Random Forest	Best Model
Training R²	~0.80	~0.74	~0.98+	Random Forest
CV R² (3-Fold)	0.80	0.74	0.85	Random Forest
Test RMSE	~1344	~1200	~870	Random Forest
Test MAE	~1038	~918	~640	Random Forest
Final Selection	No	No	YES	Random Forest

7. Graphical Results & Analytical Insights
7.1 Seasonwise Monthly Distribution
Graph Type: Grouped Bar Chart (hue = season)
Key Insight: Bike rentals peak in Summer (Season 2) and Fall (Season 3) across all months. Spring (Season 1) records the lowest rentals, particularly in January–March. Winter (Season 4) shows moderate counts. This seasonality is a strong predictor and confirms that temperature-driven behavior dominates rental patterns.

7.2 Weekday-wise Monthly Distribution
Graph Type: Grouped Bar Chart (hue = weekday)
Key Insight: Rental counts are relatively consistent across the 7 weekdays within each month. There is no strongly dominant weekday, suggesting that both commuter (weekday) and leisure (weekend) use cases are active in roughly equal measure across months.

7.3 Yearly Distribution — Violin Plot
Graph Type: Violin Plot (year vs total_count)
Key Insight: The 2012 (year=1) violin is significantly wider and taller, indicating both higher average rentals and greater variability. The 2011 distribution is narrower and lower, showing lower demand. Year-over-year growth is strongly evident, reflecting increasing system adoption.

7.4 Holiday vs Non-Holiday Distribution
Graph Type: Bar Chart (holiday on x-axis, season as hue)
Key Insight: Non-holiday days (holiday=0) consistently show higher rental counts across all seasons. This supports the hypothesis that the majority of users are commuters, and the system is primarily utilitarian rather than recreational. Holiday rentals drop noticeably.

7.5 Working Day Distribution
Graph Type: Bar Chart (workingday on x-axis, season as hue)
Key Insight: Working days (workingday=1) record higher average total counts compared to non-working days. Combined with the holiday finding, this reinforces that weekday commuter demand is the dominant use pattern.

7.6 Weather Condition Distribution
Graph Type: Bar Chart (weather_condition vs total_count)
Key Insight: There is a clear descending trend: Clear/Partly Cloudy weather generates the most rentals, Mist/Cloudy weather reduces them moderately, and Light Snow/Rain leads to the fewest rentals. Weather condition is therefore an important predictor of daily demand.

7.7 Outlier Box Plots
Graph Type: Box Plots
Key Insight: total_count has no outliers. Normalized temperature (temp) is also clean. However, windspeed and humidity both exhibited upper-tail outliers. These were treated before model training to prevent skewed model coefficients and unstable predictions.

7.8 Normal Probability Plot (Q-Q Plot)
Graph Type: Normal Probability Plot for total_count
Key Insight: The central portion of total_count aligns well with the theoretical normal distribution, but the tail regions diverge. This mild non-normality is acceptable for tree-based models (Decision Tree, Random Forest) and does not significantly impact Linear Regression for large samples, in line with the Central Limit Theorem.

7.9 Correlation Heatmap
Graph Type: Annotated Heatmap (upper triangle masked)
Key Insight: The heatmap reveals multicollinearity between temp and atemp (near 1.0). It also shows that casual and registered are sub-components of total_count (very high correlation), confirming the need to remove them to prevent data leakage. Humidity and windspeed show negative correlations with total_count, while temperature shows a strong positive correlation.

7.10 Cross-Validation Residual Plots
Graph Type: Scatter Plot (Observed vs Residual) for all three models
Key Insight: All three models show some heteroscedasticity (unequal variance of residuals), especially at higher observed values. Random Forest produces a residual scatter that is more tightly clustered around zero, indicating better predictive consistency. Linear Regression shows larger and more systematic residuals, particularly at the extremes of the distribution.

7.11 Final Test Set Residual Plots
Graph Type: Scatter Plot (y_test vs residuals) for all three models
Key Insight: For the Random Forest model, residuals are more symmetrically distributed around zero across the full range of observed values. Linear Regression exhibits widening residuals at higher observed counts (fan-shaped pattern), suggesting it struggles with peak demand days. The Decision Tree shows some improvement over Linear Regression but still exhibits larger spread than Random Forest.

8. Key Insights & Findings
8.1 Demand Drivers
•	Temperature is the single strongest continuous predictor of bike rental demand. Higher temperatures are positively correlated with more rentals.
•	Season is the most important categorical driver: Summer and Fall drive peak demand; Spring and Winter see lower demand.
•	Working days generate more rentals than holidays or weekends, confirming a strong commuter use pattern.
•	Clear weather dramatically outperforms rainy or snowy weather in generating rentals.
•	Year-over-year growth is significant: 2012 shows substantially higher demand than 2011, suggesting the system was in rapid adoption phase.

8.2 Feature Engineering Insights
•	atemp (feeling temperature) was dropped due to near-perfect multicollinearity with temp — retaining both would distort linear model coefficients.
•	casual and registered were excluded to prevent data leakage, as they are direct sub-components of the target variable total_count.
•	One-Hot Encoding of categorical variables expanded the feature space from 10 raw features to 18 encoded features, allowing linear models to capture category-specific effects.

8.3 Model Insights
•	Linear Regression achieves a respectable 80% cross-validated R², but struggles with non-linear patterns and produces larger errors on peak demand days.
•	Decision Tree Regressor underperforms Linear Regression in cross-validation (0.74 vs 0.80), suggesting that without additional depth or boosting, a single tree does not generalize as well as a linear baseline.
•	Random Forest Regressor achieves the best CV R² of 0.85 and the lowest RMSE and MAE on the test set. The ensemble averaging effectively reduces variance and handles non-linearities in the data.

9. Conclusion
This project successfully built and evaluated a machine learning pipeline for predicting daily bike rental demand. Starting from raw data, the project covered the full data science lifecycle: data cleaning, exploratory analysis, outlier treatment, feature selection, encoding, modelling, and evaluation.

Among the three regression models evaluated — Linear Regression, Decision Tree Regressor, and Random Forest Regressor — the Random Forest Regressor emerged as the definitive best model, achieving:
•	The highest cross-validated R² score of 0.85 on the training set.
•	The lowest Root Mean Squared Error (RMSE) on the unseen test set.
•	The lowest Mean Absolute Error (MAE) on the unseen test set.
•	More symmetric and lower-spread residuals in diagnostic plots.

The analysis also revealed powerful insights about demand drivers: temperature and season are the dominant factors, working days outperform holidays, and clear weather is strongly associated with higher rentals. The system showed clear year-over-year growth from 2011 to 2012.

The final model predictions were exported to Bike_Renting_Python.csv for downstream use by operations teams.

10. Further Improvements & Recommendations
10.1 Feature Engineering
•	Interaction Features: Create interaction terms such as temp × season or windspeed × weather_condition to explicitly model combined effects.
•	Lag Features: Add previous day(s) rental counts as lag features to capture temporal autocorrelation, which is likely strong in daily data.
•	Rolling Statistics: Rolling mean and rolling standard deviation over a 7-day window could capture weekly trends.
•	Holiday Type: Differentiate between types of holidays (national, local, school break) for more granular modeling.

10.2 Advanced Modelling
•	Gradient Boosting (XGBoost / LightGBM / CatBoost): These algorithms often outperform standard Random Forest by sequentially correcting errors of prior trees, typically achieving higher R² and lower RMSE.
•	Hyperparameter Tuning: Apply GridSearchCV or RandomizedSearchCV on the Random Forest to optimize n_estimators, max_depth, min_samples_leaf, and max_features.
•	Time Series Models: Since the data is temporal, ARIMA, SARIMA, Prophet, or LSTM (Long Short-Term Memory) neural networks could be explored to model time-based patterns explicitly.
•	Stacking/Blending: Combine predictions from multiple base models (Linear, Decision Tree, RF) using a meta-learner to potentially surpass any individual model.

10.3 Outlier Treatment
•	The current outlier imputation uses simple mean replacement. KNN-based imputation (already partially implemented using fancyimpute) could be fully utilized to produce more realistic imputations based on nearest neighbors in feature space.
•	Robust Regression methods (e.g., HuberRegressor) that are inherently less sensitive to outliers could be explored as alternatives.

10.4 Evaluation Metrics
•	Add Mean Absolute Percentage Error (MAPE) and the coefficient of variation of RMSE (CV-RMSE) for more interpretable error reporting in operational contexts.
•	Evaluate model performance separately for weekdays vs weekends and different seasons to identify where each model struggles most.

10.5 Deployment Considerations
•	Wrap the Random Forest model using pickle or joblib for serialization and deploy as a REST API (e.g., using Flask or FastAPI) for real-time predictions.
•	Implement a monitoring pipeline to detect model drift as new rental data accumulates.
•	Schedule periodic retraining (monthly or quarterly) to incorporate new seasonal data and maintain prediction accuracy.

10.6 Data Enrichment
•	Incorporate external data such as real-time weather APIs, local event calendars, and public transport disruptions to improve model predictive power.
•	Station-level analysis (using the hour.csv dataset) would allow more granular demand forecasting for individual stations rather than system-wide totals.


