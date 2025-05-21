import random
import pandas as pd

file_path = 'data/test.csv'

items = [
    [1,'bread'],
    [2,'jams'],
    [3,'butter'],
    [4,'cheese'],
    [5,'milk']
]

# Transaction,Item_no,Item_name,Quantity
data = []

unique_transaction = 5

for tx in range(1, unique_transaction + 1):
    chosen_items = random.sample(items, random.randint(1,5))

    for i, (item_no, item_name) in enumerate(chosen_items, start=1):
        quantity = random.randint(1, 5)
        data.append([tx, item_no, item_name, quantity])

df = pd.DataFrame(data, columns=["Transaction", "Item_no", "Item_name", "Quantity"])
# print(df.head())
    
df.to_csv(file_path, index=False)