#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

rawPivot = pd.DataFrame(pd.read_json('transaction-data-adhoc-analysis.json'))
rawPivot


# In[ ]:


#Table: Count per Month


# In[ ]:


filename = "transaction-data-adhoc-analysis.json"
rawData = open(filename,'r+')
df = pd.DataFrame(json.load(rawData))
   
month = {
        '01/':'JAN',
        '02/':'FEB',
        '03/':'MAR',
        '04/':'APR',
        '05/':'MAY',
        '06/':'JUN',
}

# for every month, look for the i in the transaction_date then we label that as the month

for i in month:
    df.loc[df['transaction_date'].str.contains(i),'month'] = month[i]

def monthly_count(mon):
    
    # arranging by month, depending on which 'mon' is in the df[mon]
    monthly = df[df.month==mon]
    sorted_df = monthly.sort_values(by=['transaction_date'])

    # item list to separate brands, products and quantities
    item_list = []
    for index,row in sorted_df.iterrows():
        product_split = (row['transaction_items'].split(';'))
        for i in range(0,len(product_split)):
            item_list.append(product_split[i].split(','))
    
    
    item_df = pd.DataFrame(item_list,columns=['BRAND','PRODUCT','QUANTITY'])
    item_df['QUANTITY'] = item_df['QUANTITY'].str.extract('(\d+)',expand=False)
    item_df['QUANTITY'] = item_df['QUANTITY'].astype(int)

    itemcount = item_df.groupby('PRODUCT').sum()
   
    return itemcount.squeeze()

net_product_sales_df = pd.DataFrame({i:monthly_count(i) for i in list(month.values())})
net_product_sales_df


# In[ ]:


#Total Sale Value per Item per Month


# In[ ]:


all_salevaluedf = df[['transaction_items','transaction_value']].drop_duplicates(subset=['transaction_items'])

unique_salevaluedf = pd.DataFrame(all_salevaluedf.loc[all_salevaluedf['transaction_items'].str.contains('x1') 
                                                    & (all_salevaluedf['transaction_items'].str.contains(';')== False)])

# concept: change the index of the column names, moving it to the appropriate column
def count_items_index(value):
    count_index = value[value.index(',')+1:value.index(',',value.index(',')+1)]
    return count_index

# applying index change
unique_salevaluedf['Product Name'] = unique_salevaluedf['transaction_items'].apply(count_items_index)

# getting cost of each item from respective product
cost = unique_salevaluedf.set_index('Product Name')['transaction_value']

# labeling cost with respective name
net_revenue = pd.DataFrame({'Cost for each product':cost})

# for each i in  the total number of i's in the keys of net_product_sales_df, 
# we multiply the i'th value of the cost of the total revenue by the i'th value of the products_sale_count_df

for i in tuple(net_product_sales_df.keys()):
    net_revenue[i] = net_revenue['Cost for each product'] * net_product_sales_df[i]
    
net_revenue = net_revenue[['Cost for each product','JAN','FEB','MAR','APR','MAY','JUN']]

net_revenue


# In[ ]:


# Bar Graph for Monthly Sold Goods


# In[ ]:


# displays quantity of items sold within the month

def bar_graph(mon):
    d = {mon : net_product_sales_df[mon]}
    mon_graph_df = pd.DataFrame(data = d)
    mon_graph_df.iloc[0:0]
    return mon_graph_df.plot(kind="barh", title="Number of Monthly Goods Sold", width=0.5)


# In[ ]:


bar_graph('JAN')


# In[ ]:


bar_graph('FEB')


# In[ ]:


bar_graph('MAR')


# In[ ]:


bar_graph('APR')


# In[ ]:


bar_graph('MAY')


# In[ ]:


bar_graph('JUN')


# In[ ]:


#Repeaters


# In[ ]:


def repeater(mon):
    month = ["JAN","FEB","MAR","APR","MAY","JUN"]
    month_index = month.index(mon)
    
    monthRpt = df[df["month"]==month[month_index]]
    prevMonthRpt = df[df["month"]==month[month_index-1]]
    
    if(mon == 'JAN'):
        return 0
    else:
        CMR = set(monthRpt["username"])
        PMR = set(prevMonthRpt["username"])
        return len(CMR & PMR)


# In[ ]:


repeater('JAN')


# In[ ]:


repeater('FEB')


# In[ ]:


repeater('MAR')


# In[ ]:


repeater('APR')


# In[ ]:


repeater('MAY')


# #Inactive Customers

# In[ ]:


# Inactive
month = ['JAN','FEB','MAR','APR','MAY','JUN']

def inactive(mon):
    month_number = month.index(mon)+1
    january = set(df[df['month'].str.contains('JAN')]['name'])
    february = set(df[df['month'].str.contains('FEB')]['name'])
    march = set(df[df['month'].str.contains('MAR')]['name'])
    april = set(df[df['month'].str.contains('APR')]['name'])
    may = set(df[df['month'].str.contains('MAY')]['name'])
    june = set(df[df['month'].str.contains('JUN')]['name'])
    notjanuary = set(df[~df['month'].str.contains('JAN')]['name'])
    notfebruary = set(df[~df['month'].str.contains('FEB')]['name'])
    notmarch = set(df[~df['month'].str.contains('MAR')]['name'])
    notapril = set(df[~df['month'].str.contains('APR')]['name'])
    notmay = set(df[~df['month'].str.contains('MAY')]['name'])
    notjune = set(df[~df['month'].str.contains('JUN')]['name'])
    
    conditions = set()
    if month_number == 1:
        pass
    elif month_number == 2:
        conditions = (january & notfebruary)
        inactive_df = {'col1': [1, 2], 'col2': [3, 4]}
    elif month_number == 3:     
        conditions = (january & february & notmarch)
    elif month_number == 4:   
        conditions = (january & february & march & notapril)
    elif month_number == 5:   
        conditions = (january & february & march & april & notmay)
    elif month_number == 6:   
        conditions = (january & february & march & april & may & notjune)
    else:
        return len(conditions)
    return len(conditions)


# In[ ]:


inactive('JAN')


# In[ ]:


inactive('FEB')


# In[ ]:


inactive('MAR')


# In[ ]:


inactive('APR')


# In[ ]:


inactive('MAY')


# In[ ]:


inactive('JUN')


# In[ ]:


#Loyal Customers


# In[ ]:


#loyal: 

month = ['JAN','FEB','MAR','APR','MAY','JUN']

def loyal(mon):
    month_number = month.index(mon)+1
    january = set(df[df['month'] == 'JAN']['name'])
    february = set(df[df['month'] == 'FEB']['name'])
    march = set(df[df['month'] == 'MAR']['name'])
    april = set(df[df['month'] == 'APR']['name'])
    may = set(df[df['month'] == 'MAY']['name'])
    june = set(df[df['month'] == 'JUN']['name'])

    conditions = set()
    if month_number == 1:
        conditions = january
    elif month_number == 2:   
        conditions = (january & february)
    elif month_number == 3:     
        conditions = (january & february & march)
    elif month_number == 4:   
        conditions = (january & february & march & april)
    elif month_number == 5:   
        conditions = (january & february & march & april & may)
    elif month_number == 6:   
        conditions = (january & february & march & april & may & june)
    else:
        pass
    if len(conditions) == 0:
        return (f'Not a valid month!')
    else:
        return len(conditions)


# In[ ]:


loyal('JAN')


# In[ ]:


loyal('FEB')


# In[ ]:


loyal('MAR')


# In[ ]:


loyal('APR')


# In[ ]:


loyal('MAY')


# In[ ]:


loyal('JUN')

