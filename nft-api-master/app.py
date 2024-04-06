import requests
import json
import csv
from flask import Flask, jsonify, request
from flask_cors import CORS
# from flask_restful import Api, Resource, reqparse, abort, fields, marshal_with
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.models import load_model


app = Flask(__name__)
CORS(app)
# api = Api(app)


@app.route('/predictions/contract_address=<contr_add>&no_of_months=<no_months>', methods=['GET'])
def return_pred_price(contr_add, no_months):
    # Validate the contract address
    pattern = r'^0x[0-9a-fA-F]{40}$'
    if not re.match(pattern, contr_add):
        return jsonify({'error': 'Invalid contract address'})
  
    with open('collections.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        found = False
        for row in reader:
            if row[0] == contr_add:
                found = True
                timestamp_min = row[1]
                x_train_min =  row[2]
                x_train_max = row[3]
                break

        if found == False:
            return jsonify({'error': 'model does not exist yet for this contract address'})
     
    last4char = contr_add[-4:]
    timestamp_min = datetime.strptime(timestamp_min, "%Y-%m-%d %H:%M:%S")

    # print('here: ',timestamp_min, type(timestamp_min))# <class 'str'>

    n_months_from_now = datetime.now() + timedelta(days=30 * int(no_months))
    n_months_time_diff = (n_months_from_now - timestamp_min) / pd.Timedelta(days=1)
    # print("nmonths time diff =",n_months_time_diff, type(n_months_time_diff))    
   
    model = load_model(f'{last4char}.h5')   

    # Scale the input for prediction
    input_data = np.array([[n_months_time_diff]])
    # input_data_scaled = scaler.transform(input_data)
    input_data_scaled = (input_data - float(x_train_min)) / (float(x_train_max) - float(x_train_min))

    print("scaled data= ", input_data_scaled, input_data_scaled.shape)
    
    # Reshape the input data for prediction
    input_data_reshaped = np.reshape(input_data_scaled, (input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))

    # Make the prediction
    predicted_price = model.predict(input_data_reshaped)

    print(f"Predicted price after {no_months} months: {float(predicted_price[0][0])}")

    return jsonify({'Predicted Price': float(predicted_price[0][0])})
    

@app.route('/train_model/contract_address=<contr_add>', methods=['GET'])
def train_save(contr_add):
    # Validate the contract address
    pattern = r'^0x[0-9a-fA-F]{40}$'
    if not re.match(pattern, contr_add):
        return jsonify({'error': 'Invalid contract address'})   
     
    # API request using user input contract address
    url = f"https://ethereum-rest.api.mnemonichq.com/collections/v1beta2/{contr_add}/prices/DURATION_365_DAYS/GROUP_BY_PERIOD_1_DAY"
    headers = {'x-api-key': 'dIyltLFEZBck7lk5QHvvk0if70cM0kpTCfy1I4Oa7nPGpi5h'}
    result = requests.get(url, headers=headers).json()
    df = pd.DataFrame(result)

    # Convert timestamp to datetime format
    df["timestamp"] = df.apply(lambda x: pd.to_datetime(x[0]["timestamp"]), axis=1)

    # Convert avg to numeric values
    df["avg"] = df.apply(lambda x: pd.to_numeric(x[0]["avg"]), axis=1)

    # Calculate the time difference from the first timestamp
    df["time_diff"] = (df["timestamp"] - df["timestamp"].min()) / pd.Timedelta(days=1)

    # Prepare the data for training
    train_data = df[["time_diff", "avg"]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:, :-1], train_data.iloc[:, -1], test_size=0.2, shuffle=False)

    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # print("here",type(X_train))
    X_test_scaled = scaler.transform(X_test)

    # Reshape the input data for LSTM
    X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, X_train_scaled.shape[1]), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=50, batch_size=1, verbose=2)

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(None)

    # Calculate the minimum and maximum values for each feature in X_train
    x_train_min = float(X_train.min(axis=0))
    x_train_max = float(X_train.max(axis=0))

    timestamp_min = df["timestamp"].min()

    # Open a CSV file for writing
    with open('collections.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow([contr_add, timestamp_min, x_train_min, x_train_max])

    last4char = contr_add[-4:]
    
    #save model as h5 file
    model.save(f'{last4char}.h5')    

    return jsonify({'messgae': f'Model for {contr_add} successfully trained & saved'})
    


# api.add_resource(TestClass, "/next")    

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
