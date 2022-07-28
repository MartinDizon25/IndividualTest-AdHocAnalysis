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


#Table: Items sold per Month


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
    
monthly_df = rawPivot
monthly_df['transaction_date'] = pd.to_datetime(monthly_df['transaction_date'])

#get all unique monthly periods that cover the entire dataset
month_year = monthly_df['transaction_date'].dt.strftime('%Y/%m').unique().tolist()
#Get the first day of each month
first_day_of_month = pd.to_datetime(month_year)
#Get the last day of each month
last_day_of_month = first_day_of_month.map(lambda e: e.replace(day=calendar.monthrange(e.year, e.month)[1]))

period_values = {
    'period': month_year, #The 'period' column is the month in 'YYYY/mm format
    'period_start': first_day_of_month, #'The period_start' is the first day of the month
    'period_end': last_day_of_month #The 'period_end' is the last day of the month
}
    
periods = pd.DataFrame(period_values)
    
    
#Add 'period' column
for index, row in periods.iterrows():
#Filter transactions to a specific period
    filtered_trans = monthly_df[(monthly_df['transaction_date'] >= row['period_start'])
                                                        & (monthly_df['transaction_date'] <= row['period_end'])]
    monthly_df.loc[filtered_trans.index, 'period'] = row['period']
    
#Convert period to 'YYYY/mm' format
monthly_df['period'] = monthly_df['period'].dt.strftime('%Y/%m')
    
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


#Table: Total Revenue per Item per Monthle Value per Item per Month


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


# Table and Bar Graph for Repeat Customers


# In[ ]:


def repeater(showGraph):
    month = ["JAN","FEB","MAR","APR","MAY","JUN"]
    def rpt(mon):
        month_index = month.index(mon)
        
        #Finds number of customers in current month
        month_rpt = df[df["month"]==month[month_index]]
        #Finds number of customers in previous month
        prev_month_rpt = df[df["month"]==month[month_index-1]]
    
        if(mon == 'JAN'):
            #Sets initial repeat customers to 0
            return 0
        else:
            CMR = set(month_rpt["username"])
            PMR = set(prev_month_rpt["username"])
            #Finds number of customers who shopped in previous month
            return len(CMR & PMR)
    
    d = [rpt('JAN'), rpt('FEB'), rpt('MAR'), rpt('APR'), rpt('MAY'), rpt('JUN')]
    rpt_graph_df = pd.DataFrame(index = month)
    rpt_graph_df['Repeat Customers'] = d
    if(showGraph):
        return rpt_graph_df.plot(kind="line", title="Number of Repeat Customers", figsize=(8, 4))
    else:
        return rpt_graph_df


# In[ ]:


repeater('False')


# In[ ]:


repeater('True')


# In[ ]:


# Table and Bar Graph for Inactive Customers


# In[ ]:


def inactive(showGraph):
    periods = pd.DataFrame(period_values)
    def inactiveEntry(row, monthly_df):
        #Filters the dataset before the period
        before_period = df[monthly_df['transaction_date'] < row['period_start']]
    
        #Generates the list of unique customers who have transactions until the previous month
        customers_before = before_period['username'].unique()
    
        within_period = df[(monthly_df['transaction_date'] >= row['period_start']) & (monthly_df['transaction_date'] <= row['period_end'])]
    
        #Generates the list of unique customers who have transactions within the current month
        customers_within = within_period['username'].unique()
    
        #Removes customers from the first list that also made purchase(s) within the period
        no_duplicate_customers = [cust for cust in customers_before if cust not in customers_within]
    
        #Returns the number of customers with past transactions
        return(len(no_duplicate_customers))

    inactive_num = periods
    inactive_num['Inactive Customers'] = inactive_num.apply(lambda row: inactiveEntry(row, monthly_df), axis=1)
    inactive_num.drop(columns=['period', 'period_start', 'period_end'], inplace=True)
    inactive_num = inactive_num.rename(index={0: 'JAN', 1: 'FEB', 2: 'MAR', 3: 'APR', 4: 'MAY',5: 'JUN'})
    if(showGraph):
        return inactive_num.plot(kind="line", title="Number of Inactive Customers", figsize=(8, 4))
    else:
        return inactive_num


# In[ ]:


inactive('False')


# In[ ]:


inactive('True')


# In[ ]:


# Table and Bar Graph for Engaged Customers


# In[ ]:


def engaged(showGraph):
    periods = pd.DataFrame(period_values)
    def engagedEntry(monthly_df, periods):
        
        customers_per_month = []
    
        for index, row in periods.iterrows():
            #Selects the start of the month
            curr_period_start = periods.loc[index, 'period_start']
            #Selects the end of the month
            curr_period_end = periods.loc[index, 'period_end']
            #Selects all transactions within the month
            curr_period_transactions = df[(monthly_df['transaction_date'] >= curr_period_start) 
                                  & (monthly_df['transaction_date'] <= curr_period_end)]
            #Selects users from the current month and prevents duplicates 
            curr_month_customers = set(curr_period_transactions['username'].unique())
            customers_per_month.append(curr_month_customers)
    
        #Gets the unique customers with transactions for each month
        #Get the common customers from the first month up to the current month
        #This will determine the unique users who have transactions every single month until the current month.
        #Note the on the first month, the value of 'engaged_count' is simply the count of unique customers during that month
        engaged_customers = customers_per_month[0]
        for index, row in periods.iterrows():        
            if index != 0:
                engaged_customers = engaged_customers.intersection(customers_per_month[index])
    
            #Store the result in 'engaged_count' column
            periods.loc[index, 'Engaged Customers'] = len(engaged_customers)
    
    
        #Convert the values under 'repeaters_count' to integer since it is expressed in float    
        periods['Engaged Customers'] = periods['Engaged Customers'].astype(np.int64)
        return periods
        
    engaged_num = engagedEntry(monthly_df, periods)
    engaged_num.drop(columns=['period', 'period_start', 'period_end'], inplace=True)
    engaged_num = engaged_num.rename(index={0: 'JAN', 1: 'FEB', 2: 'MAR', 3: 'APR', 4: 'MAY',5: 'JUN'})
    #engaged_num = engaged_num.rename(index={0: 'JAN', 1: 'FEB', 2: 'MAR', 3: 'APR', 4: 'MAY',5: 'JUN'})
    if(showGraph):
        return engaged_num.plot(kind="line", title="Number of Engaged Customers", figsize=(8, 4))
    else:
        return engaged_num


# In[ ]:


engaged(False)


# In[ ]:


engaged(True)

