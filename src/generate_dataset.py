import random
import pandas as pd

file_path = 'data/test.csv'

items = [
    [1, 'bread'],
    [2, 'jams'],
    [3, 'butter'],
    [4, 'cheese'],
    [5, 'milk'],
    [6, 'eggs'],
    [7, 'cereal'],
    [8, 'pasta'],
    [9, 'rice'],
    [10, 'flour'],
    [11, 'sugar'],
    [12, 'salt'],
    [13, 'pepper'],
    [14, 'tea'],
    [15, 'coffee'],
    [16, 'orange juice'],
    [17, 'apples'],
    [18, 'bananas'],
    [19, 'lettuce'],
    [20, 'tomatoes'],
    [21, 'potatoes'],
    [22, 'onions'],
    [23, 'chicken'],
    [24, 'beef'],
    [25, 'fish']
]


# Transaction,Item_no,Item_name,Quantity
data = []

unique_transaction = 1234

for tx in range(1, unique_transaction + 1):
    chosen_items = random.sample(items, random.randint(1,len(items)))

    for i, (item_no, item_name) in enumerate(chosen_items, start=1):
        quantity = random.randint(1, 5)
        data.append([tx, item_no, item_name, quantity])

df = pd.DataFrame(data, columns=["Transaction", "Item_no", "Item_name", "Quantity"])
# print(df.head())
    
df.to_csv(file_path, index=False)