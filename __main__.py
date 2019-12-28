import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main(): 
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


    table = np.array([date, consumption, coal, gas, hidroelectric, nuclear, wind, solar, biomass])
    table = np.transpose(table)


    table = remove_abberant_data(table, [coal, gas, hidroelectric, nuclear, wind, solar, biomass], [2,3,4,5,6,7,8])
    
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_columns = scaler.fit_transform(table[:,[2,3,4,5,6,7,8]])
    table = table.astype('float64')
    table[:,[2,3,4,5,6,7,8]] = scaled_columns
    
    #primele 20 de randuri dupa punctele b) si c)
    for i in range (0,20):
        for j in range(0,len(table[0])):
            print(table[i][j], end="    ")
        print()

    
    
    

def remove_abberant_data(table, columns_to_check, index_columns):
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