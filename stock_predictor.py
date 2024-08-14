import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Set up the Streamlit app
st.title("Stock Analysis App")
st.write("Enter a stock ticker symbol and analyze its performance.")

# Input for stock ticker
stock_ticker = st.text_input("Enter the stock ticker symbol (e.g., 'AAPL' for Apple):", "AAPL")

# Load the data for the specified stock ticker
if stock_ticker:
    st.write(f"Loading data for {stock_ticker}...")
    data = yf.download(stock_ticker, start="2010-01-01", end="2023-12-31")
    plt.figure(figsize = (16,8))
    plt.plot(data['Close'], label = 'Closing Price')
    st.pyplot(plt)
    if not data.empty:
        st.write("Data loaded successfully!")
        
        # Display the raw data
        st.write("### Raw Data")
        st.write(data.tail())

        # Preprocess the data
        data['Open - Close'] = data['Open'] - data['Close']
        data['High - Low'] = data['High'] - data['Low']
        data = data.dropna()

        # Define features and target
        X = data[['Open - Close', 'High - Low']]
        st.write(X.head())
        Y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

        # Normalize the features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)


        data['Open - Close'] = data['Open'] - data['Close']

        data['High - Low'] = data['High'] - data['Low']

        data  = data.dropna()


        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score

# Using gridsearch to find the best parameter
        params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
        knn = neighbors.KNeighborsClassifier()
        model = GridSearchCV(knn, params, cv=5)

# Fit the model
        model.fit(X_train, y_train)

# Accuracy Score
        accuracy_train = accuracy_score(y_train, model.predict(X_train))
        accuracy_test = accuracy_score(y_test, model.predict(X_test))

        st.write('Train_data Accuracy: %.2f' % accuracy_train)
        st.write('Test_data Accuracy: %.2f' % accuracy_test)


        prediction_classification  = model.predict(X_test)
        actual_predicted_data = pd.DataFrame({'Actual Class': y_test, 'Predicted Class': prediction_classification})

        st.write(actual_predicted_data)

        y = data['Close']


        from sklearn.neighbors import KNeighborsRegressor
        from sklearn import neighbors
        from sklearn.model_selection import train_test_split

        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state = 44)

# Using gridsearch to find the best parameter
        params = {'n_neighbors':[2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
        knn_reg = neighbors.KNeighborsRegressor()
        model_reg = GridSearchCV(knn_reg, params, cv=5)

# Fit the model and make predictions
        model_reg.fit(X_train_reg, y_train_reg)
        predictions = model_reg.predict(X_test_reg)
        st.write("Predictions:")
        st.write(predictions)

        # RMSE (Root Mean Square Error)
        rms = np.sqrt(np.mean(np.power((np.array(y_test_reg) - np.array(predictions)), 2)))
        rms

        valid = pd.DataFrame({'Actual Close': y_test_reg, 'Predicted Close value': predictions})

        valid.head(10)



    # Implementing k-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=44)

# Random Forest Classifier
        rf_clf = RandomForestClassifier(random_state=44)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_rf_clf = GridSearchCV(rf_clf, param_grid, cv=kf, scoring='accuracy')
        grid_rf_clf.fit(X_train, y_train)

        # Best parameters and score
        best_params = grid_rf_clf.best_params_
        best_score = grid_rf_clf.best_score_

        st.write("Best parameters found: ", best_params)
        st.write("Best cross-validated score: {:.2f}".format(best_score))

        # Evaluate the best model on the test set
        best_model = grid_rf_clf.best_estimator_
        test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
        st.write("Test Accuracy of the best model: {:.2f}".format(test_accuracy))

# Evaluate the classifier
        y_pred_train = grid_rf_clf.predict(X_train)
        y_pred_test = grid_rf_clf.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        st.write(f'Random Forest Classifier - Train Accuracy: {train_accuracy:.2f}')
        st.write(f'Random Forest Classifier - Test Accuracy: {test_accuracy:.2f}')



        rf_reg = RandomForestRegressor(random_state=44)
        param_grid_reg = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_rf_reg = GridSearchCV(rf_reg, param_grid_reg, cv=kf, scoring='neg_mean_squared_error')
        grid_rf_reg.fit(X_train, y_train)




        # Evaluate the regressor
        y_pred_train_reg = grid_rf_reg.predict(X_train)
        y_pred_test_reg = grid_rf_reg.predict(X_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_reg))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_reg))
        st.write(f'Random Forest Regressor - Train RMSE: {train_rmse:.2f}')
        st.write(f'Random Forest Regressor - Test RMSE: {test_rmse:.2f}')

        # Analyze predictions
        valid = pd.DataFrame({'Actual Close': y_test, 'Predicted Close value': y_pred_test_reg})
        st.write(valid.head(10))

        # Plot Actual vs Predicted Close Values
        plt.figure(figsize=(16,8))
        plt.plot(valid['Actual Close'], label='Actual Close')
        plt.plot(valid['Predicted Close value'], label='Predicted Close', linestyle='--')
        plt.legend()
        st.pyplot(plt)



    else:
        st.write("No data found for the given ticker symbol.")
