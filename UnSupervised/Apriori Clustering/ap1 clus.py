import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('store_data.csv')  #//Importing Dataset from system

print(store_data.head())   #// to check the header

store_data = pd.read_csv('store_data.csv', header=None)  #/keeping header as None
print(store_data.head())

records = []
for i in range(0, 7501):
    records.append([str(store_data.values[i,j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

print(len(association_results)) # //to check the Total Number of Rules mined
print(association_results[0])  #// to print the first item the association_rules list to see the first rule

for item in association_results:                                         # // to display the rule, the support, the confidence, and lift for each rule in a more clear way:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
