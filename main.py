import pandas as pd
from mlxtend.frequent_patterns import association_rules
import ast
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth

#تعداد محصولات یکتا در کل دادگان

#read Dataset
dataset=pd.read_csv('data/supermaket_edit.csv')

# remove noise and Data cleaning
# dataset['Product'] = dataset['Product'].apply(lambda x: re.sub('[\/\,\.\(\)\-]','',x))
# dataset=dataset.drop_duplicates()
# print(dataset[dataset.duplicated(keep=False)])
# dataset.to_csv('data/supermaket_edit.csv') # Save Edit Data


print("Number product in market.. ",dataset['Product'].nunique())
# print("List product in market.. ",dataset['Product'].unique())

print("Number Customer Id.. ",dataset['Customer Id'].nunique())
# print("List product in Customer Id.. ",dataset['Customer Id'].unique())

num_Date=dataset['Date'].nunique()
print("Number Date.. ",num_Date)
# print("List product in Date.. ",dataset['Date'].unique())

###################################################################
#	میزان فروش به ازای هر روز به صورت میانگین چند است

dS_grouped = dataset.groupby('Date',sort=True).count()
sum_Date=dS_grouped['Product'].sum()
print("Average sell.. ",round((sum_Date/num_Date),2))

###################################################################
#	کدام روز هفته، بیشترین تعداد محصول فروش رفته‌است؟

Max_date=dataset['Date'].value_counts(ascending=True)
# print("Maximum Day Sell..",dS_grouped['Product'].max())
print("Frequently product in basket..",Max_date.index[-1],Max_date.iloc[-1])

###################################################################
#پنج مشتری‌ای که در سال ۲۰۲۰، بیشترین تعداد "سبد" را داشته‌اند، کدامند

# Customer_grouped=dataset.groupby(['Customer Id','Date']).sum()
# Customer_grouped=dataset.groupby(['Customer Id','Date'])['Product'].agg(lambda x: ast.literal_eval(str(list(x)))).reset_index(name='Products')
import json

Customer_grouped = dataset.groupby(['Customer Id', 'Date'])['Product']\
    .agg(lambda x: json.loads(json.dumps(list(x)))) \
    .reset_index(name='Products')
# Customer_grouped.to_csv('data/grouped_data.csv')

basket=pd.read_csv('data/grouped_data.csv')
basket['Date'] = pd.to_datetime(basket['Date'])
basket2020 = basket[basket['Date'].dt.year == 2020]
Freq_basket=basket2020['Customer Id'].value_counts()
for i in range(5):
    print(f"{i+1} Frequently basket CustomerId ={Freq_basket.index[i]} , Frequent={Freq_basket.iloc[i]}")

###################################################################
#	چهار محصولی که کمتر از بقیه در سبد مشتریان قرار گرفته‌اند، کدامند
print("Frequently product in basket..",dataset['Product'].value_counts(ascending=True).head(4))

###################################################################


encoded_baskets = []
te = TransactionEncoder()
sds=basket['Products']
for Pbasket in sds:
    aa = ast.literal_eval(Pbasket)
    encoded_baskets.append(aa)

#frequent itemset
te_ary = te.fit(encoded_baskets).transform(encoded_baskets)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)

# sort result
sorted_patterns = frequent_itemsets.sort_values(by='support', ascending=False)
top_5_products = sorted_patterns.head(5)
print(top_5_products)

#create association rules from frequent_itemset
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.0)
sorted_rules = rules.sort_values(by='confidence', ascending=False)
top_2_rules = sorted_rules.head(2)
# max confidence
print(top_2_rules)

#print All association rules
for index, row in rules.iterrows():
    print("Rule:", row['antecedents'], "->", row['consequents'])
    print("Support:", row['support'])
    print("Confidence:", row['confidence'])
    print("Lift:", row['lift'])
    print("\n")
