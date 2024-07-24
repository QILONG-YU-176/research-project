import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pywt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Attention,Conv1D, MaxPooling1D, Flatten, RepeatVector, TimeDistributed, Input
from keras.layers import InputLayer
from keras.layers import concatenate, Attention, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from sklearn.preprocessing import LabelEncoder
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, average_precision_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate, Input, Attention, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.layers import InputLayer, RepeatVector, TimeDistributed
from sklearn.preprocessing import LabelEncoder
import pywt
import os
import tweepy
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap
#####################
# Load data
file_path = '/Users/yuqilong/Desktop/data/nflx_2014_2023 2.csv'
#file_path = '/Users/yuqilong/Desktop/data/tsla_2014_2023.csv'


# Read the csv file
def read_csv_file(file_path):
    return pd.read_csv(file_path)

# Read CSV file
data = read_csv_file(file_path)


# Calculated logarithmic return
data['return'] = np.log(data['next_day_close'] / data['close']) * 100

# denoising
def denoise_wavelet(data):
    coeffs = pywt.wavedec(data, 'db4', level=5)
    threshold = 0.5 * np.sqrt(2*np.log(len(data)))
    coeffs_threshold = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    data_denoised = pywt.waverec(coeffs_threshold, 'db4')
    return data_denoised

data['return_denoised'] = denoise_wavelet(data['return'])


#Sentiment labels

import nltk
nltk.download('vader_lexicon')



# Set environment variables
os.environ['CONSUMER_KEY'] = 'kVrfTqKXlcFGYjfd09JcgkIC0'
os.environ['CONSUMER_SECRET'] = 'bAqMIPNxy224WsHyWw7b6qC8fFk8wOKgtyS34CfzJ0ye9pm9Qd'
os.environ['ACCESS_TOKEN'] = '1787960570148159488-CntQn0G7QEyJD3YhCfHkZGyptpfpaC'
os.environ['ACCESS_SECRET'] = 'j2bDICvjP2Y4UUP9TppJ5JnEUJsBHRRM36z0GtLb9SQVE'
os.environ['BEARER_TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAKTCtgEAAAAA7jPTJl2OtfHiMyN1FgA560zq7dE%3D4B1N9I2CavOQbf2xZDvQavnEzEFEQ1mzjgApggcBrqgoX2xjxS'

# Access environment variables
consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_secret = os.getenv('ACCESS_SECRET')
bearer_token = os.getenv('BEARER_TOKEN')

# Authenticate to Twitter API v1.1
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

# Authentication with Twitter API v2 using Bearer Token
client = tweepy.Client(bearer_token)

# Search for tweets
query = 'netflix stock -is:retweet'
tweets = client.search_recent_tweets(query=query, max_results=100, tweet_fields=["created_at", "text"])

# Process tweets if found
if tweets.data:
    data = pd.DataFrame({
        'tweet': [tweet.text for tweet in tweets.data],
        'date': [tweet.created_at for tweet in tweets.data]
    })

    sia = SentimentIntensityAnalyzer()
    data['sentiment'] = data['tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

    data['date'] = pd.to_datetime(data['date'])
    daily_sentiment = data.groupby(data['date'].dt.date)['sentiment'].mean()
    print(daily_sentiment)
else:
    print("No tweets found.")


# Year, month and day of withdrawal
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# Extract the day of the week
data['day_of_week'] = data['date'].dt.dayofweek  # 星期一是0，星期天是6

# The first days of the year
data['day_of_year'] = data['date'].dt.dayofyear

# Week of the month
data['week_of_month'] = data['date'].apply(lambda d: (d.day-1) // 7 + 1)

# Is it the last month of the quarter
data['is_quarter_end'] = data['date'].dt.is_quarter_end

# Is it the end of the year
data['is_year_end'] = data['date'].dt.is_year_end

# ‘holidays’
import holidays
us_holidays = holidays.US()
data['is_holiday'] = data['date'].apply(lambda x: x in us_holidays)


#Drawing the raw data
# Plot original data
plt.figure(figsize=(20, 6))
plt.plot(data['date'], data['return_denoised'], lw=2)
plt.xlabel("Date", fontsize=16)
plt.ylabel(" Logarithmic return (%)", fontsize=16)
plt.title("NFLX  Logarithmic return value", fontsize=16)
plt.show()


# Divide the training and test sets
train_data = data.query('date <  "2022-01-01"').reset_index(drop=True)
test_data = data.query('date >= "2022-01-01"').reset_index(drop=True)

#Remove redundant variables
X_train = train_data.drop(['date','return', 'return_denoised', "next_day_close",'label','label1','label2'], axis=1)
y_train = train_data['return_denoised']


X_test = test_data.drop(['date', 'return', 'return_denoised', "next_day_close",'label','label1','label2'], axis=1)
y_test = test_data['return_denoised']



#Draw the training and test sets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 6))
ax1.plot(train_data.date, train_data['return_denoised'], lw=2)
ax1.set_xlabel("date", fontsize=16)
ax1.set_ylabel("Logarithmic return value(%)", fontsize=16)
ax1.set_title("Training data", fontsize=16)
ax2.plot(test_data.date, test_data['return_denoised'], c='orange', lw=2)
ax2.set_xlabel("date", fontsize=16)
ax2.set_ylabel("Logarithmic return value(%)", fontsize=16)
ax2.set_title("Test data", fontsize=16);
plt.show()

#normalisation
# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Leave the target variables unchanged
y_train_scaled = y_train.values.reshape(-1, 1)
y_test_scaled = y_test.values.reshape(-1, 1)

# Reshape input data to include time dimension
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))




####################################

# modelling
input_layer = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

# convolutional layer
from keras.layers import Dropout
cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
#cnn = Conv1D(filters=16, kernel_size=1, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
cnn = MaxPooling1D(pool_size=2)(cnn)
#cnn = Dropout(0.05)(cnn)
cnn = Flatten()(cnn)

#LSTM layer
lstm = LSTM(50, return_sequences=True)(input_layer)
#lstm = LSTM(25, return_sequences=True, kernel_regularizer=l2(0.01))(input_layer)
lstm = Dropout(0.05)(lstm)
attention = Attention()([lstm,lstm])
#attention = Dropout(0.05)(attention)
attention = Flatten()(attention)

#Merge Layer
combined = concatenate([cnn, attention])
#combined = Dense(10, activation='linear', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(combined)

#output layer
output_layer = Dense(1,activation='linear')(combined)

model = Model(inputs = input_layer, outputs = output_layer)
from keras.optimizers import Adam

model.compile(optimizer='adam', loss = "mean_squared_error")
model.summary()

# Setting the early stop parameters
early_stopping = EarlyStopping(
    monitor='val_loss',         # 监控模型的验证损失
    min_delta=0.001,            # 停止前损失至少减少0.001
    patience=100,                # 如果10个epochs内val_loss没有改善，则提前停止训练
    verbose=1,                  # 打印早停的信息
    mode='min',                 # 监控损失，所以模式是'min'
    restore_best_weights=True   # 恢复到最佳模型权重
)

# Fit the model using reshaped input data
history = model.fit(X_train_reshaped, y_train_scaled,epochs=1000, batch_size=32,validation_data=(X_test_reshaped, y_test_scaled),callbacks=[early_stopping])


# Prediction using models
y_pred_scaled = model.predict(X_test_reshaped)

y_predtrain_scaled = model.predict(X_train_reshaped)

# inverse normalisation
y_true = test_data['return_denoised']

y_truetrain= train_data['return_denoised']

y_true_scaled = y_true.values.reshape(-1, 1)

y_truetrain_scaled = y_truetrain.values.reshape(-1, 1)



# Define the threshold range
threshold_range = 1

# Convert probabilities to binary predictions based on the threshold range
binary_pred = [(1 if abs(pred - true) <= threshold_range else 0) for pred, true in zip(y_pred_scaled , y_true_scaled )]

# Convert continuous y_true to binary based on threshold range
binary_true = [(1 if abs(true_value-true_value) <= threshold_range else 0) for true_value in y_true_scaled]

# Compute Accuracy within threshold range
accuracy_within_range = accuracy_score(binary_true, binary_pred)

# Compute F1 Score within threshold range
f1_within_range = f1_score(binary_true, binary_pred)

# Compute PR AUC within threshold range
pr_auc_within_range = average_precision_score(binary_true, binary_pred)

#roc_auc_within_range = roc_auc_score(binary_true, binary_pred)

print("Accuracy within range:", accuracy_within_range)
print("F1 Score within range:", f1_within_range)
print("PR AUC within range:", pr_auc_within_range)

# Array of true and predicted values
true_values = np.array([y_true_scaled,y_true_scaled])  # 替换为你的真实值数组
predicted_values = np.array([y_pred_scaled,y_true_scaled])  # 替换为你的预测值数组

# Calculate the mean value
mean_true = np.mean(true_values)
mean_predicted = np.mean(predicted_values)
# Reshape true_values and predicted_values if they have more than 2 dimensions

# Reshape true_values and predicted_values if they have more than 2 dimensions
true_values = np.squeeze(true_values)
predicted_values = np.squeeze(predicted_values)

# Compute the Pearson correlation coefficient
pearson_corr = np.corrcoef(true_values, predicted_values)[0, 1]

# Print the Pearson correlation coefficient
print("Pearson Correlation Coefficient:", pearson_corr)


mse = np.mean((y_true_scaled - y_pred_scaled)**2)
print("Mean Squared Error (MSE):", mse)



# Calculate R^2 score for test data
r2_test = r2_score(y_true_scaled, y_pred_scaled)
print("R^2 Score for Test Data:", r2_test)


# Calculate residuals for both training and testing sets
train_residuals = y_truetrain_scaled - y_predtrain_scaled
test_residuals = y_true_scaled - y_pred_scaled




# Plotting true versus predicted values
plt.figure(figsize=(6,6))
plt.plot(train_data.date,  y_truetrain_scaled, c='orange',label='True Values')
plt.plot(train_data.date, y_predtrain_scaled, lw=3, c='r',linestyle = '-', label='Predictions')
plt.legend(loc="lower left")
plt.xlabel("Date", fontsize=16)
plt.ylabel(" Logarithmic return (U.S. dollar)", fontsize=16)
plt.title("NFLX  Logarithmic return value (training data)", fontsize=16)
plt.show()

# Plotting true versus predicted values
plt.figure(figsize=(6,6))
plt.plot(test_data.date, y_true_scaled , c='orange',label='True Values')
plt.plot(test_data.date, y_pred_scaled , lw=3, c='r',linestyle = '-', label='Predictions')
plt.legend(loc="lower left")
plt.xlabel("Date", fontsize=16)
plt.ylabel("Logarithmic return (U.S. dollar)", fontsize=16)
plt.title("NFLX  Logarithmic return value (testing data)", fontsize=16)
plt.show()

# Plot the residual plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_predtrain_scaled, train_residuals, c='blue', label='Training data')
plt.scatter(y_pred_scaled, test_residuals, c='orange', label='Testing data')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()

# Plot the residual density plot
plt.subplot(1, 2, 2)
plt.hist(train_residuals, bins=20, density=True, alpha=0.5, color='blue', label='Training data')
plt.hist(test_residuals, bins=20, density=True, alpha=0.5, color='orange', label='Testing data')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Residual Density Plot')
plt.legend()
plt.show()

# Plot the loss curve with both training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()


# sharp

# building SHAP explainer
class RandomForestWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        # 使用模型预测功能
        return self.model.predict(X)


# Instantiating Model Wrappers
model_wrapper = RandomForestWrapper(model)





# Convert non-floating point numbers to floating point numbers
X_test= X_test.astype(float)  # 将所有列转换为 float 类型



# Now use SHAP to create the interpreter
explainer = shap.Explainer(model_wrapper,X_test)  # 假设 X_train 是训练数据

# Interpreting test data using an interpreter
shap_values = explainer(X_test)
# Mapping SHAP Aggregates
shap.summary_plot(shap_values,X_test)

shap.plots.bar(shap_values)

#shap.plots.text(shap_values[2])

shap.plots.heatmap(shap_values[:1000])


# Assuming the previous steps are followed and the data is preprocessed as shown in your original code

#

# Creating the LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    mode='regression',
    feature_names=X_train.columns,
    discretize_continuous=True
)

# Choosing an instance from the test set to explain
i = 0  # Index of the instance to explain
exp = explainer.explain_instance(X_test.iloc[i], model.predict, num_features=10)

# Displaying the explanation
exp.show_in_notebook(show_all=False)

# Aggregate LIME explanations for multiple instances
n_instances = 100  # Number of instances to explain
lime_explanations = []

for i in range(n_instances):
    exp = explainer.explain_instance(X_test.iloc[i], model.predict, num_features=10)
    lime_explanations.append(exp)

# Summarize LIME explanations
def summarize_lime_explanations(lime_explanations):
    feature_importance = {}

    for exp in lime_explanations:
        for feature, importance in exp.as_list():
            feature_name = feature.split('=')[0].strip()
            if feature_name not in feature_importance:
                feature_importance[feature_name] = 0
            feature_importance[feature_name] += importance

    sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_importance

summary = summarize_lime_explanations(lime_explanations)

# Selecting top 10 features
top_10_summary = summary[:10]
features, importances = zip(*top_10_summary)

# Plotting the summary
plt.figure(figsize=(10, 5))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 LIME Feature Importance Summary')
plt.show()

# Define bull and bear markets based on a threshold
threshold = 0.5  # Example threshold for market conditions
bull_market = data[data['return_denoised'] > threshold]
bear_market = data[data['return_denoised'] < -threshold]

# Feature selection and scaling
features = ['Daily Sentiment (News)', 'Daily Sentiment (Financial data provider)', 'Daily Sentiment (Social media)','macd']  # Replace with actual feature names
target = 'return_denoised'

# Clean data and drop NaN values
bull_market_cleaned = bull_market.dropna(subset=features + [target])
X_bull = bull_market_cleaned[features]
y_bull = bull_market_cleaned[target]

bear_market_cleaned = bear_market.dropna(subset=features + [target])
X_bear = bear_market_cleaned[features]
y_bear = bear_market_cleaned[target]

# Scale the features
scaler = StandardScaler()
X_bull = scaler.fit_transform(X_bull)
X_bear = scaler.transform(X_bear)




####################################

from sklearn.metrics import accuracy_score
# Predict and evaluate on bull market data
y_pred_bull = model.predict(X_bull)
mse_bull = mean_squared_error(y_bull, y_pred_bull)
r2_bull = r2_score(y_bull, y_pred_bull)


# Predict and evaluate on bear market data
y_pred_bear = model.predict(X_bear)

mse_bear = mean_squared_error(y_bear, y_pred_bear)
r2_bear = r2_score(y_bear, y_pred_bear)

print(f'Bull Market - MSE: {mse_bull},,R2: {r2_bull}')
print(f'Bear Market - MSE: {mse_bear}, R2: {r2_bear}')

# Function to perform sensitivity analysis
def sensitivity_analysis(model, X, y, param_variations):
    results = {}
    for param, values in param_variations.items():
        for value in values:
            X_copy = X.copy()
            X_copy[param] = value  # Adjust the parameter value
            X_scaled = scaler.transform(X_copy)  # Use the previously fitted scaler
            y_pred = model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            results[f'{param}_{value}'] = mse
    return results

# Define parameter variations
param_variations = {
    'Daily Sentiment (News)': [-1, 0, 1],  # Example variations
    'Daily Sentiment (Financial data provider)': [-1, 0, 1],
    'Daily Sentiment (Social media)': [-1, 0, 1],
    'macd':[-0.5, 0.5,1]
}

# Perform sensitivity analysis on bull market model
sensitivity_bull = sensitivity_analysis(model, bull_market[features].dropna(), y_bull, param_variations)

# Perform sensitivity analysis on bear market model
sensitivity_bear = sensitivity_analysis(model, bear_market[features].dropna(), y_bear, param_variations)

print('Sensitivity Analysis - Bull Market:', sensitivity_bull)
print('Sensitivity Analysis - Bear Market:', sensitivity_bear)

# Plot performance metrics
metrics = ['MSE', 'R2']
values_bull = [mse_bull, r2_bull]
values_bear = [mse_bear, r2_bear]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].bar(metrics, values_bull, color='green')
ax[0].set_title('Bull Market Performance')
ax[0].set_ylabel('Metric Value')

ax[1].bar(metrics, values_bear, color='red')
ax[1].set_title('Bear Market Performance')
ax[1].set_ylabel('Metric Value')

plt.show()

# Plot sensitivity analysis results
sensitivity_df_bull = pd.DataFrame(sensitivity_bull.items(), columns=['Parameter', 'MSE'])
sensitivity_df_bear = pd.DataFrame(sensitivity_bear.items(), columns=['Parameter', 'MSE'])

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='Parameter', y='MSE', data=sensitivity_df_bull, ax=ax[0])
ax[0].set_title('Sensitivity Analysis - Bull Market')
ax[0].set_ylabel('MSE')

sns.barplot(x='Parameter', y='MSE', data=sensitivity_df_bear, ax=ax[1])
ax[1].set_title('Sensitivity Analysis - Bear Market')
ax[1].set_ylabel('MSE')

plt.show()
