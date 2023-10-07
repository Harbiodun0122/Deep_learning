# fit_transform should only be done to training data, only transform should be done to the test data
# how to identify an outlier in a regression problem
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

income_class = pd.read_csv(
    r'C:\Users\Harbiodun\Desktop\Phone\Download\Deep Learning\Tensorflow\What is Tensorflow\census_data_What is Tensorflow.csv')
print(income_class.head())
# print(income_class.isna().sum())

categorical_columns = ['workclass', 'education', 'occupation', 'native_country']
numerical_colums = ['education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# display a histogram of the age, just to check the highest age bracket in labor
income_class['age'].hist(bins=20)
# plt.show()

# dropping colums that are not needed
income_class = income_class.drop(['age', 'marital_status', 'relationship', 'race', 'gender'], axis=1)

# drop rows that has the value '?' in any of it's column
def drop_row(column: list):
    global income_class
    for i in column: # loop through the list
        income_class = income_class.drop(income_class[income_class[i] == ' ?'].index, axis=0)
    income_class = income_class.reset_index() # reset the index of the dataframe
    income_class = income_class.drop(['index'], axis=1) # drop the unnecessary 'index' column created
    return income_class

drop_row(['workclass', 'occupation', 'native_country'])
print(income_class)


education_num = []
capital_gain = []
capital_loss = []
hours_per_week = []

def min_max(dataframe):
    for i in dataframe['education_num']:
        education_num.append(i)

    for i in dataframe['capital_gain']:
        capital_gain.append(i)

    for i in dataframe['capital_loss']:
        capital_loss.append(i)

    for i in dataframe['hours_per_week']:
        hours_per_week.append(i)

    edu_num = 'EDUCATION NUM: min: ', min(education_num), 'max: ', max(education_num)
    cap_gain = 'CAPITAL GAIN: min: ', min(capital_gain), 'max: ', max(capital_gain)
    cap_loss = 'CAPITAL LOSS: min: ', min(capital_loss), 'max: ', max(capital_loss)
    hrs_per_week = 'HOURS PER WEEK: min: ', min(hours_per_week), 'max: ', max(hours_per_week)
    return edu_num, cap_gain, cap_loss, hrs_per_week

# splitting the income_class dataset into the x_labels(features) and y_label(target)
x_labels = income_class.drop('income_bracket', axis=1)
y_label = income_class['income_bracket']
print(x_labels.shape)
print(y_label.shape)

# scale down the numerical columns so that the models won't be biased
# x_train_test_columns = ['education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'workclass', 'education', 'occupation', 'native_country']
# col_trans = ColumnTransformer([('Scaler', MinMaxScaler(), numerical_colums)], remainder='passthrough')
# x_labels = pd.DataFrame(col_trans.fit_transform(x_labels), columns=x_train_test_columns)
# print(x_labels)
# print(y_label)

# change the value of the y_label dataset
# if value is <=50K value should be 1 else value should be 0
def fix_label(value):
    if value.upper() == ' <=50K':
        return 0
    else:
        return 1

y_label = y_label.apply(fix_label)
# print(y_label)

# adding each column to feature columns in tensorflow
# numeric column
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

# categorical column with hash bucket
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=1000)
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', hash_bucket_size=1000)

feat_cols = [education_num, capital_gain, capital_loss, hours_per_week, workclass, education, occupation, native_country]
# print('x_labels: \n', x_labels)
# split into training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x_labels, y_label, test_size=0.25, random_state=0)
print('x_train: \n', x_train)
print('x_test: \n', x_test)
print(y_train)
print(y_test)

input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=32, num_epochs=100, shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)#, optimizer='Adam')#, model_dir='class income')
model.train(input_fn=input_func, steps=100)

# making predictions
pred_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=32, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
print(list(predictions)[:50])

# evaluating the model
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
print(results)