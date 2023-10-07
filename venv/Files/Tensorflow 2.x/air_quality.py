import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime, warnings, scipy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from IPython.core.interactiveshell import InteractiveShell
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# setting up the graphics
plt.rcParams['patch.force_edgecolor'] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor='dimgray', linewidth=1)

InteractiveShell.ast_node_interactivity = 'last_expr'
pd.options.display.max_columns = 50
warnings.filterwarnings('ignore')

# reading the dataset from the air_quality.csv
df = pd.read_csv(r'datasets\air_quality.csv', sep=' ')
print(f'Printing out the dataframe\n{df.head()}')

# convert time to datetime format
def combine_date(df, tab_name):
    list_tab = []
    for i in range(df.shape[0]):
        list_tab.append(df.loc[i, 'Tanggal'] + 'T' + df.loc[i, tab_name][0:2]) #[0:2]
    return np.array(list_tab, dtype='datetime64')

df['Datetime'] = combine_date(df, 'Jam')
print(f'Dataframe after combining date and time\n{df.head()}')

# convert into hourly data
df = df[['Datetime', 'O3', 'CO', 'NO2', 'SO2', 'NO', 'CO2', 'VOC', 'PM1', 'PM2.5', 'PM4', 'PM10', 'TSP', 'TEMP', 'HUM', 'WS', 'WD', 'ISPU']]
df2 = df.groupby(['Datetime']).mean()
print(f'Dataframe after grouping into hourly data\n{df.head()}')
print(f'Dataframe2 after grouping into hourly data\n{df2.head()}')
print(f'Dataframe2 after grouping into hourly data description\n{df2.describe}')


# calculate quantile for each variable, THIS IS SAME AS DOING DF2.DESCRIBE
def calculate_quantile(i, df2):
    Q1 = df2[[i]].quantile(0.25)[0]
    Q3 = df2[[i]].quantile(0.75)[0]
    IQR = Q3 - Q1
    min = df2[[i]].min()[0]
    max = df2[[i]].max()[0]
    min_IQR = Q1 - 1.5*IQR
    max_IQR = Q3 + 1.5*IQR

    return Q1, Q3, min, max, min_IQR, max_IQR

# it's just to show how the thing really works, what i'm printing
# print('df2.iloc[[1736]]: \n', df2.iloc[[1736]])
# print('df2.index[0]: \n', df2.index[0])
# print('df2.shape: \n', df2.shape)
# print('df.index[df2.shape[0]-1]: \n', df.index[df2.shape[0]-1])
# delete first and last rows to avoid missing value extrapolation
df2.drop(index=[df2.index[0], df2.index[df2.shape[0]-1]], inplace=True)
print(f'Dataframe2 after dropping the first and last rows\n{df2.head()}')

# find and interpolate the outliers
for i in df2.columns:
    print('\nAttribute-', i, ':')
    Q1, Q3, min, max, min_IQR, max_IQR = calculate_quantile(i, df2)
    print(f'Q1 =  {Q1}')
    print(f'Q3 =  {Q3}')
    print(f'min_IQR =  {min_IQR}')
    print(f'max_IQR =  {max_IQR}')
    if (min < min_IQR):
        print(f'----> Low outlier is found = {min}')
    if (max < max_IQR):
        print(f'----> High outlier is found = {max}')

    def convert_nan(x, max_IQR=max_IQR, min_IQR=min_IQR):
        if ((x > max_IQR) | (x < min_IQR)):
            x = np.nan
        else:
            x = x
        return x

    def convert_nan_HUM(x, max_IQR=100.0, min_IQR=min_IQR):
        if ((x > max_IQR) | (x < min_IQR)):
            x = np.nan
        else:
            x = x
        return x

    # This is just cleanup, nothing much
    # the humidity column
    if (i == 'HUM'):
        df2[i] = df2[i].map(convert_nan_HUM)
        df2[i] = df2[i].interpolate(method='linear')
    # other columns
    if (i != 'HUM'):
        df2[i] = df2[i].map(convert_nan)
        df2[i] = df2[i].interpolate(method='linear')
    if (len(df2[df2[i].isnull()][i]) == 0):
        print('#################### Outliers have been interpolated ####################')

# log transformation to deal with skewed data
dataset = np.log1p(df2[['TEMP']].values)
print(dataset.shape)

# create new dataframe to compare the original vs log transform data
dist_df = pd.DataFrame({'TEMP' : df2['TEMP'].values, 'log_TEMP': dataset[:,0]})
print('dist_df.head(): \n', dist_df.head())

# Histogram plot original vs log transform data
plt.figure(figsize=(12, 5))
dist_df.hist()
plt.show()

# split the dataset into train and test sets, for me i'll be using train_test_split
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print('len(train), len(test): ', len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape the train and test data
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print('trainX: \n', trainX)
print('trainY: \n', trainY)
print('Shape of trainX: ', trainX.shape)
print('Shape of trainY: ', trainY.shape)
print('Shape of testX: ', testX.shape)
print('Shape of testY: ', testY.shape)

# reshape the input array in the form ~ [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print('trainX: \n', trainX)
print('trainY: \n', trainY)
print('Shape of trainX: ', trainX.shape)
print('Shape of testX: ', testX.shape)

# create and fit the LSTM Neural Network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
# Dense is the standard neural network, which by default uses the relu activation function
# Dense is almost used in every model, it is the basic foward propagation - reverse propagation error
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=250, batch_size=32, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
print('trainPredict: \n', trainPredict)
testPredict = model.predict(testX)
print('testPredict: \n', testPredict)

# invert predictions
trainPredict = np.expm1(trainPredict)
print('inverted trainPredict: \n', trainPredict)
trainY = np.expm1(trainY)
print('inverted trainY: \n', trainY)
testPredict = np.expm1(testPredict)
print('inverted testPredict: \n', testPredict)
testY = np.expm1(testY)
print('inverted testY: \n', testY)

# calculate root mean squred error
# if the RMSE of test score is better than the RMSE of  training score, that means we've overtrained our model OR
# if the RMSE of test score is better than the STD of test score, that means we've overtrained our model
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
print(f'Train Score: {trainScore}')
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print(f'Test Score: {testScore}')

# JUST FOR A NICE LITTLE PRESENTATION
test_series = pd.Series(testY)
# State of model performance
if testScore < test_series.std():
    print('\n [Model performance is GOOD enough]')
    print('\nRMSE of test prediction < Standard deviation of test dataset')
    print(f'{testScore} < {test_series.std()}')
else:
    print('\n [Model performance is NOT GOOD enough]')
    print('\nRMSE of test prediction > Standard deviation of test dataset')
    print(f'{testScore} > {test_series.std()}')

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot original dataset and make predictions
time_axis = np.linspace(0, dataset.shape[0]-1, 15)
time_axis = np.array([int(i) for i in time_axis])
time_axisLab = np.array(df2.index, dtype='datetime64[D]')

fig = plt.figure()
ax = fig.add_axes([0, 0, 2.1, 2])
ax.plot(np.expm1(dataset), label='Original Dataset')
ax.plot(trainPredictPlot, color='orange' ,label='Train Prediction')
ax.plot(testPredictPlot, color='red', label='Test Prediction')
ax.set_xticks(time_axis)
ax.set_xticklabels(time_axisLab[time_axis], rotation=45)
ax.set_xlabel('\nDate', fontsize=27, fontweight='bold')
ax.set_ylabel('ISPU', fontsize=27, fontweight='bold')
ax.legend(loc='best', prop={'size':20})
ax.tick_params(size=10, labelsize=15)
ax.set_xlim([-1, 1735])
# plt.show()

ax1 = fig.add_axes([2.3, 1.3, 1, 0.7])
ax1.plot(np.expm1(dataset), label='Original Dataset')
ax1.plot(testPredictPlot, color='red', label='Test Prediction')
ax1.set_xticks(time_axis)
ax1.set_xticklabels(time_axisLab[time_axis], rotation=45)
ax1.set_xlabel('Date', fontsize=27, fontweight='bold')
ax1.set_ylabel('ISPU', fontsize=27, fontweight='bold')
ax1.tick_params(size=10, labelsize=15)
ax1.set_xlim([1360, 1735])
plt.show()