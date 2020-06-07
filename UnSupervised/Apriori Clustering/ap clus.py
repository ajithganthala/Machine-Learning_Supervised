import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori, association_rules 

# Loading the dt 
dt = pd.read_excel('retail.xlsx') 
print(dt.head()) 

# Exploring the columns of the dt 
print(dt.columns)

# Exploring the different regions of transactions 
print(dt.Country.unique())

# Stripping extra spaces in the description 
dt['Description'] = dt['Description'].str.strip() 

# Dropping the rows without any invoice number 
dt.dropna(axis = 0, subset =['InvoiceNo'], inplace = True) 
dt['InvoiceNo'] = dt['InvoiceNo'].astype('str') 

# Dropping all transactions which were done on credit 
dt = dt[~dt['InvoiceNo'].str.contains('C')]

# Transactions done in France 
basket_France = (dt[dt['Country'] =="France"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done in the United Kingdom 
basket_UK = (dt[dt['Country'] =="United Kingdom"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

# Transactions done in Portugal 
basket_Por = (dt[dt['Country'] =="Portugal"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo')) 

basket_Sweden = (dt[dt['Country'] =="Sweden"] 
		.groupby(['InvoiceNo', 'Description'])['Quantity'] 
		.sum().unstack().reset_index().fillna(0) 
		.set_index('InvoiceNo'))

# Defining the hot encoding function to make the data suitable 
# for the concerned libraries 
def hot_encode(x): 
	if(x<= 0): 
		return 0
	if(x>= 1): 
		return 1

# Encoding the datasets 
basket_encoded = basket_France.applymap(hot_encode) 
basket_France = basket_encoded 

basket_encoded = basket_UK.applymap(hot_encode) 
basket_UK = basket_encoded 

basket_encoded = basket_Por.applymap(hot_encode) 
basket_Por = basket_encoded 

basket_encoded = basket_Sweden.applymap(hot_encode) 
basket_Sweden = basket_encoded 

# Building the model 
frq_items = apriori(basket_France, min_support = 0.05, use_colnames = True) 

# Collecting the inferred rules in a dataframe 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head())

frq_items = apriori(basket_UK, min_support = 0.01, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 

frq_items = apriori(basket_Por, min_support = 0.05, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 

frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True) 
rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 
print(rules.head()) 





