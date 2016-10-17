import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV, BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam
import numpy as np
import csv
import matplotlib.pyplot as plt


### load data

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
df = pd.concat([train, test])

print "Train:", train.shape
print "Test: ", test.shape


### convert loss to log of loss (optional -- this won't apply for all models)

# train["loss"] = np.log(train["loss"])


### feature encoding

ntrain = train.shape[0]
ntest = test.shape[0]

df = pd.get_dummies(df)

train = df.iloc[:ntrain,:]
test = df.iloc[ntrain:,:]


### remove outliers (optional -- generally better results on neural net w/ this applied)

print "Train before removing outliers", train.shape

train = train[train.loss < 25000]

print "Train after removing outliers", train.shape


### normalization (optional)

# (Allstate's data looks like it's potentially already been normalized, and results for several models were better w/out this step)

# cat_cols = [c for c in df.columns if "cat" in c]
# cont_cols = [c for c in df.columns if "cont" in c]
# all_cols = cat_cols + cont_cols

# sc = StandardScaler()   # alt options: minmax, robustscaler, no normalization

# for col in all_cols:
#     sc.fit(df[col].values)
#     train[col] = sc.transform(train[col].values)
#     test[col] = sc.transform(test[col].values)


### create feature matrix and target vector

X = train.drop(["id", "loss"], axis=1).as_matrix()
y = np.array(train["loss"].values)


### Feature reduction (optional)

sel = VarianceThreshold()

sel.fit(X, y)

print "Train before removing low-variance features", X.shape

X = sel.transform(X)

print "Train after removing low-variance features", X.shape


### define models and hyperparameters

lr = LinearRegression()
br = BayesianRidge()
net = ElasticNetCV(l1_ratio=[.1, .7, .95, .99, 1], normalize=False)
rf = RandomForestRegressor(n_estimators=75)


### build neural net model

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=2,
                               mode="auto")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=7)

model = Sequential()
model.add(Dense(2048, input_shape=(X_train.shape[1],), init='uniform', activation='relu'))

model.add(Dropout(0.0))
model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.0))
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.0))
model.add(Dense(1024, activation='relu'))

model.add(Dense(1))

optimizer = Adam(lr=.005)
model.compile(loss='mae', optimizer=optimizer)

model.fit(X_train, y_train,
          nb_epoch=25,
          batch_size=756,
          validation_data=[X_val, y_val],
          verbose=0,
          callbacks=[early_stopping],
         )


print 'Validation MAE w/ Neural Net:  {0}'.format(mean_absolute_error(y_val, model.predict(X_val)))


# ^^ the above parameters were chosen through a (by no means exhaustive) grid search of model architecture and hyperparameters. Code for that step below


### gridsearch

# Note: This takes a really f-ing long time, even with GPU, CnMem, and CuDNN enabled


# def create_model(neurons_input=1024, neurons_one=512, neurons_two=512, neurons_three=64, dropout_rate=0.0,
#     optimizer='adam', activation="relu", init_mode="uniform", lr=0.001, verbose=0):
#     model = Sequential()
#     model.add(Dense(neurons_input, input_shape=(X.shape[1],), init=init_mode, activation=activation))

#     model.add(Dropout(dropout_rate))
#     model.add(Dense(neurons_one, activation=activation))

#     model.add(Dropout(dropout_rate))
#     model.add(Dense(neurons_two, activation=activation))

#     model.add(Dropout(dropout_rate))
#     model.add(Dense(neurons_three, activation=activation))

#     model.add(Dense(1))

#     model.compile(loss='mae', optimizer=optimizer)
#     return model

# model = KerasRegressor(build_fn=create_model, verbose=0)

# param_grid = dict(nb_epoch=[6],
#                   batch_size=[756],
#                   lr=[.005],
#                   dropout_rate=[0.0],
#                   neurons_input=[2048],
#                   neurons_one=[1024],
#                   neurons_two=[128],
#                   neurons_three=[1024],
#                   )

# grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=100, scoring='mean_absolute_error')
# grid_result = grid.fit(X, y)

# print "best params:", grid.best_params_
# print "best score: ", grid.best_score_


### ensemble predictions

regressors = [br, lr, net, model, rf]


def ensemble_regressor(rgrs, data):
    """computes the mean prediction for each sample across models"""
    if data.ndim == 1:        # for single sample predictions
        preds = []
        for rgr in rgrs:
            if rgr != model:  # keras model prediction inputs are formatted slightly different
                preds.append(rgr.predict(data)[0])
            else:
                reshaped_data = data.reshape(1, data.shape[0])
                preds.append(rgr.predict(reshaped_data)[0][0])
        return np.mean(preds)

    preds = pd.DataFrame(columns=range(data.shape[0]))   # for batch predictions
    for rgr in rgrs:
        if rgr == model:
            preds = preds.append(pd.DataFrame([[p for p in rgr.predict(data)]], columns=range(data.shape[0])), ignore_index=True)
        else:
            preds = preds.append(pd.DataFrame([rgr.predict(data)], columns=range(data.shape[0])), ignore_index=True)
    return [preds[col].mean() for col in preds.columns.values] if rgr != model else [preds[col].mean()[0] for col in preds.columns.values]


### training and cross-validation

kf = KFold(n_splits=3)

for train_split, test_split in kf.split(X):
    br.fit(X[train_split], y[train_split])
    lr.fit(X[train_split], y[train_split])
    net.fit(X[train_split], y[train_split])
    rf.fit(X[train_split], y[train_split])

    print 'BayesianRidge MAE:  {0}'.format(mean_absolute_error(y[test_split], br.predict(X[test_split])))
    print 'LinReg MAE:         {0}'.format(mean_absolute_error(y[test_split], lr.predict(X[test_split])))
    print 'NeuralNet MAE:      {0}'.format(mean_absolute_error(y[test_split], model.predict(X[test_split])))   # not refitting NN each fold for the sake of time
    print 'ElasticNet MAE:     {0}'.format(mean_absolute_error(y[test_split], net.predict(X[test_split])))
    print 'RandomForest MAE:   {0}'.format(mean_absolute_error(y[test_split], rf.predict(X[test_split])))
    print 'Ensemble MAE:       {0}'.format(mean_absolute_error(y[test_split], ensemble_regressor(regressors, X[test_split])))
    print
    print "Current Benchmark:   X"   # I like to keep track of the current best performing score here for reference
    print


 ### visualize spread between predictions and target values

pred = np.zeros_like(y)
pred[:y.shape[0]] = [x for x in model.predict(X)]        # to visualize NN preds
# pred[:y.shape[0]] = ensemble_regressor(regressors, X)  # to visualize ensemble preds

fig, ax = plt.subplots()
ax.scatter(y, pred, c='k')
ax.plot([21000, 0], [21000, 0], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
plt.show()


### write predictions to file

ids = test["id"].values

# predictions = ensemble_regressor([regressors], test.drop(["id"], axis=1).as_matrix())                   # ensemble
# predictions = ensemble_regressor([model], test.drop(["id", "loss"], axis=1).as_matrix())                # neural net
predictions = ensemble_regressor([model], sel.transform(test.drop(["id", "loss"], axis=1).as_matrix()))   # w/ feat sel


# # w/ log loss
# with open("prediction.csv", "w") as f:
#     p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
#     for i, p in enumerate(predictions):
#         p_writer.writerow([ids[i], np.exp(p)])

# w/o log loss
# with open("prediction.csv", "w") as f:
#     p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
#     for i, p in enumerate(predictions):
#         p_writer.writerow([ids[i], p])
