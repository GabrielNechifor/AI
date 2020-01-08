import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def main(): 
    #get train data
    current_directory = os.path.dirname(sys.modules['__main__'].__file__)
    file_location = os.path.join(current_directory, "data", "train_electricity.xlsx")

    excell = pd.read_excel(file_location)

    date = list(excell['Date'])
    consumption = list(excell['Consumption_MW'])
    coal = list(excell['Coal_MW'])
    gas = list(excell['Gas_MW'])
    hidroelectric = list(excell['Hidroelectric_MW'])
    nuclear = list(excell['Nuclear_MW'])
    solar = list(excell['Solar_MW'])
    wind = list(excell['Wind_MW'])
    biomass = list(excell['Biomass_MW'])
    production = list(excell['Production_MW'])


    table = np.array([date, consumption, coal, gas, hidroelectric, nuclear, wind, solar, biomass, production])
    table = np.transpose(table)

    table = remove_aberrant_data(table, [coal, gas, hidroelectric, nuclear, wind, solar, biomass], [2,3,4,5,6,7,8])
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_columns = scaler.fit_transform(table[:,[2,3,4,5,6,7,8]])
    table = table.astype('float64')
    table[:,[2,3,4,5,6,7,8]] = scaled_columns
    
    #primele 20 de randuri dupa punctele b) si c)
    '''
    for i in range (0,20):
        for j in range(0,len(table[0])):
            print(table[i][j], end="    ")
        print()
    '''

    #define the neural network
    input_train = table[:,[2,3,4,5,6,7,8]]
   
    output_train = table[:,1]
    
    model = Sequential()
    model.add(Dense(11, input_dim=7, activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1, activation = 'linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(input_train, output_train, epochs=10, batch_size=128, verbose=0)

    #compare predictions for the same rows in the train data
    '''
    predictions = model.predict(input_train)
    for i in range(20):
	    print('%d (expected %d)' % (predictions[i], output_train[i]))
    '''
    
    #print MSE
    '''
    def baseline_model():
        model = Sequential()
        model.add(Dense(11, input_dim=7, kernel_initializer='normal', activation='relu'))
        model.add(Dense(7, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation = 'linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=128, verbose=0)
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, input_train, output_train, cv=kfold)
    print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    '''

    #get test data and predict the comsuption
    current_directory = os.path.dirname(sys.modules['__main__'].__file__)
    file_location = os.path.join(current_directory, "data", "test_electricity.xlsx")

    excell = pd.read_excel(file_location)

    coal = list(excell['Coal_MW'])
    gas = list(excell['Gas_MW'])
    hidroelectric = list(excell['Hidroelectric_MW'])
    nuclear = list(excell['Nuclear_MW'])
    solar = list(excell['Solar_MW'])
    wind = list(excell['Wind_MW'])
    biomass = list(excell['Biomass_MW'])

    test_data = np.array([coal, gas, hidroelectric, nuclear, wind, solar, biomass])
    test_data = np.transpose(test_data)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_columns = scaler.fit_transform(test_data)
    test_data = test_data.astype('float64')
    test_data = scaled_columns

    prediction = model.predict(test_data)

    #print first 20 predictions
    for i in range(20):
        print("%s => %d" % (test_data[i], prediction[i]))





def remove_aberrant_data(table, columns_to_check, index_columns):
    quartile_columns = list()
    for column in columns_to_check:
        column.sort()
        median = get_median(column)
        size = len(column)

        size_quartile = size//2

        q1_elements = column[:size_quartile]
        q1 = get_median(q1_elements)

        q3_elements = column[-size_quartile:]
        q3 = get_median(q3_elements)

        iqr = q3-q1

        quartile_columns.append([q1,q3,iqr])

    count = 0
    lines_to_delete = set()
    for index_line in range(0, len(table)):
        for index in range(0, len(index_columns)):

            index_column = index_columns[index]
            element = table[index_line][index_column]
            q1, q3, iqr = quartile_columns[index]
            
            if element < q1 - 1.5*iqr or \
                element > q3 + 1.5*iqr:
                count += 1
                lines_to_delete.add(index_line)

    table = np.delete(table, list(lines_to_delete), 0)
    return table

    
def get_median(my_list):
    my_list.sort()
    number_instances = len(my_list)
    if number_instances %2 == 1:
        median = my_list[number_instances//2]
    else:
        median = (my_list[number_instances//2] + my_list[number_instances//2 + 1]) // 2
    return median




main()