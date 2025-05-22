import pandas as pd
from itertools import combinations

class AprioriProcessor:
    """A class to process transactional data for the Apriori algorithm.
    """
    def __init__(self, path: str, unique_column: str, group_column: str, sample: float = None):
        """Initialize the AprioriProcessor with a dataset path.

        Args:
            path (str): Path to the CSV file containing transactional data.
            unique_column (str): Name of the column to be used for identifying unique transactions
            sort_column (str): Name of the column to be used for grouping
        """
        self.path = path
        self.df = pd.read_csv(self.path)
        self.threshold = None
        self.unique_column = unique_column
        self.group_column = group_column
        self.set_threshold()
        print(len(self.df))
        
        if sample is not None:
            self.df = self.df.sample(frac=sample)
            print(len(self.df))
        
        # Dataframe
        # Get unique transaction-item pairs
        self.df_transactions = self.df[[unique_column, group_column]].drop_duplicates()
        # Group items by transaction
        self.grouped = self.df_transactions.groupby(unique_column)[group_column].apply(set)
              
                
    def get_transaction_count(self):
        """Calculates the number of unique transactions in the dataset.

        Returns:
            int: Total number of unique Transaction values.
        """
        
        total_transactions = self.df[self.unique_column].nunique()
        
        return total_transactions

    def set_threshold(self, value:float = 0.2):
        """Sets the minimum support threshold based on a percentage of total transactions.

        Args:
            value (float): Support threshold as a percentage (e.g., 0.2 for 20%). Defaults to 0.2.
        """
        self.threshold: int = round(value * self.get_transaction_count())
        
    def generate_fp_itemset(self, df: pd.DataFrame = None, k: int = 1):
        
        if df is None:
            # Initial case — level 1 (1-itemsets)
            df_unique = self.df_transactions

            # Group by item and count support
            item_support = df_unique.groupby(self.group_column)[self.unique_column].nunique().reset_index()
            item_support.columns = ['Item_name', 'Support_Count']

            # Filter by threshold
            item_support = item_support[item_support['Support_Count'] >= self.threshold]
            print(f"Level {k} Frequent Itemsets:\n", item_support)

            # Recursive call with filtered 1-itemsets
            self.generate_fp_itemset(item_support, k + 1)
        
        else:
            if len(df) < 2:
                return  # No further combinations possible

            # Create set of frequent items from previous level
            frequent_items = df['Item_name'].tolist()

            # Generate k-item candidates
            candidate_sets = list(combinations(frequent_items, k))

            # Count support for each candidate set
            support_data = []
            for candidate in candidate_sets:
                count = sum(1 for items in self.grouped if set(candidate).issubset(items))
                if count >= self.threshold:
                    support_data.append((candidate, count))

            # If no candidates meet support, stop
            if not support_data:
                return

            # Create DataFrame for next iteration
            next_df = pd.DataFrame(support_data, columns=['Itemset', 'Support_Count'])
            print(f"\nLevel {k} Frequent Itemsets:\n", next_df)

            # Prepare for recursion by exploding itemsets into flat list
            exploded_items = sorted(set(item for itemset in next_df['Itemset'] for item in itemset))
            next_input_df = pd.DataFrame({'Item_name': exploded_items})

            self.generate_fp_itemset(next_input_df, k + 1)
        
if __name__ == '__main__':
    apriori = AprioriProcessor('data/e-commerce.csv', 
                               unique_column='TransactionNo',
                               group_column='ProductName',
                               sample=0.3)
    
    apriori.set_threshold(0.008)
    print(apriori.threshold)
    apriori.generate_fp_itemset()