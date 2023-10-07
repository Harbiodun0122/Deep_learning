print('loading libraries...........')
import pandas as pd
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
print('finished loading libraries')

print('readng the csv file...')
diabetes = pd.read_csv('../datasets/diabetes.csv')
print(diabetes.head())
print(diabetes.columns)

# cleaning and scaling the data
col_trans = ColumnTransformer([('Scaler', MinMaxScaler(), ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin',
                                                           'BMI', 'DiabetesPedigreeFunction'])], remainder='passthrough')
diabetes = pd.DataFrame(col_trans.fit_transform(diabetes), columns=diabetes.columns)
# already used the minmaxscaler for this
# cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#                 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
# diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda  x: (x - x.min()) / (x.max()) - x.min())
print('-------- New diabetes DataFrame --------')
print(diabetes.head())

print('Adding each column to feature columns in tensorflow...')
num_preg = tf.feature_column.numeric_column('Pregnancies')
plasma_gluc = tf.feature_column.numeric_column('Glucose')
dias_press = tf.feature_column.numeric_column('BloodPressure')
tricep = tf.feature_column.numeric_column('SkinThickness')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age = tf.feature_column.numeric_column('Age')
print('Finished adding coloumns to tensorflow')

# the next line is only in the tutorial dataset and not the dataset i'm using
# assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
diabetes['Age'].hist(bins=20)
print('Displaying age histogram...')
plt.show()

age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, age_buckets]

x_data = diabetes.drop('Outcome', axis=1)
labels = diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)
print('x_train: \n', X_train)
print('x_test: \n', X_test)
print('y_train: \n', y_train)
print('y_test: \n', y_test)

# this is just adding a layer to the neural network
# tf.compat.v1.estimator.inputs.pandas_input_fn it tensorflow 1.x feature
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=input_func,steps=1000)

# predictions
pred_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
print(list(predictions))

# evaluating the model
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(eval_input_func)
print(results)