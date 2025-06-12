import pandas as pd
from typing import List, Dict, Optional, DefaultDict, FrozenSet, Set, Tuple
from collections import defaultdict, Counter
from time import perf_counter

class FPNode:
    """Node in the FP-tree structure."""
    
    def __init__(self, item: str, count: int = 0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: Dict[str, 'FPNode'] = {}
        self.node_link = None  # Link to next node with same item
    
    def increment(self, count: int = 1):
        """Increment the count of this node."""
        self.count += count
    
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.item is None
    
    def __repr__(self):
        return f"FPNode(item={self.item}, count={self.count})"

class FPTree:
    """FP-tree data structure for efficient pattern mining."""
    
    def __init__(self):
        self.root = FPNode(None)  # Root node
        self.header_table: Dict[str, List[FPNode]] = defaultdict(list)
        self.item_counts: Dict[str, int] = {}
    
    def insert_transaction(self, transaction: List[str], count: int = 1):
        """Insert a transaction into the FP-tree."""
        current_node = self.root
        
        for item in transaction:
            if item in current_node.children:
                current_node.children[item].increment(count)
            else:
                # Create new node
                new_node = FPNode(item, count, current_node)
                current_node.children[item] = new_node
                
                # Add to header table
                self.header_table[item].append(new_node)
            
            current_node = current_node.children[item]
    
    def get_paths_for_item(self, item: str) -> List[Tuple[List[str], int]]:
        """Get all paths ending with the given item."""
        paths = []
        
        # Find all nodes with this item
        for node in self.header_table.get(item, []):
            path = []
            count = node.count
            current = node.parent
            
            # Traverse up to root, collecting items
            while current and not current.is_root():
                path.append(current.item)
                current = current.parent
            
            if path:  # Only add non-empty paths
                paths.append((path[::-1], count))  # Reverse to get correct order
        
        return paths
    
    def print_tree(self, node=None, level=0):
        """Print the FP-tree structure (for debugging)."""
        if node is None:
            node = self.root
        
        if not node.is_root():
            print("  " * level + f"{node.item}:{node.count}")
        
        for child in node.children.values():
            self.print_tree(child, level + 1)

class FPGrowthProcessor:
    """An implementation of the FP-Growth algorithm for frequent itemset mining."""
    
    def __init__(self, path: str, unique_column: str, group_column: str, sample: Optional[float] = None):
        """Initialize the FPGrowthProcessor with a dataset.

        Args:
            path (str): Path to the CSV file containing transactional data.
            unique_column (str): Name of the column for unique transactions
            group_column (str): Name of the column for items/products
            sample (float, optional): Fraction of data to sample (0-1)
        """
        self.path = path
        self.unique_column = unique_column
        self.group_column = group_column
        self.threshold = 0
        
        # Load and preprocess data
        self._load_data(sample)
        self._preprocess_transactions()
        
        # Initialize results storage
        self.frequent_itemsets = {}
        self.item_support = {}
        
    def _load_data(self, sample: Optional[float] = None):
        """Load and optionally sample the dataset."""
        self.df = pd.read_csv(self.path)
        print(f'Original dataset size: {len(self.df)}')
        
        if sample is not None and 0 < sample <= 1:
            self.df = self.df.sample(frac=sample, random_state=42)
            print(f'Sampled dataset size: {len(self.df)}')
    
    def _preprocess_transactions(self):
        """Preprocess data into transaction format."""
        # Remove duplicates and create transaction sets
        df_clean = self.df[[self.unique_column, self.group_column]].drop_duplicates()
        
        # Group items by transaction ID
        self.transactions = df_clean.groupby(self.unique_column)[self.group_column].apply(list).tolist()
        self.transaction_count = len(self.transactions)
        
        print(f'Total unique transactions: {self.transaction_count}')
    
    def set_threshold(self, min_support: float = 0.2):
        """Set minimum support threshold.
        
        Args:
            min_support (float): Support threshold as percentage (0-1)
        """
        if not 0 < min_support <= 1:
            raise ValueError("min_support must be between 0 and 1")
            
        self.min_support = min_support
        self.threshold = max(1, round(min_support * self.transaction_count))
        print(f'Minimum support threshold: {self.threshold} transactions ({min_support*100:.1f}%)')
    
    def _get_frequent_items(self) -> Dict[str, int]:
        """Get frequent 1-itemsets and their support counts."""
        item_counts = Counter()
        
        # Count all items
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        frequent_items = {item: count for item, count in item_counts.items() 
                         if count >= self.threshold}
        
        return frequent_items
    
    def _sort_transaction_by_frequency(self, transaction: List[str], frequent_items: Dict[str, int]) -> List[str]:
        """Sort transaction items by frequency (descending order)."""
        # Filter out infrequent items and sort by frequency
        frequent_in_transaction = [item for item in transaction if item in frequent_items]
        return sorted(frequent_in_transaction, key=lambda x: frequent_items[x], reverse=True)
    
    def _build_fp_tree(self, transactions: List[List[str]], frequent_items: Dict[str, int]) -> FPTree:
        """Build FP-tree from transactions."""
        fp_tree = FPTree()
        fp_tree.item_counts = frequent_items
        
        for transaction in transactions:
            # Sort transaction by item frequency
            sorted_transaction = self._sort_transaction_by_frequency(transaction, frequent_items)
            
            if sorted_transaction:
                fp_tree.insert_transaction(sorted_transaction)
        
        return fp_tree
    
    def _mine_patterns(self, fp_tree: FPTree, alpha: List[str] = None) -> Dict[frozenset, int]:
        """Mine frequent patterns from FP-tree using FP-Growth algorithm."""
        if alpha is None:
            alpha = []
        
        patterns = {}
        
        # Get items sorted by frequency (ascending order for bottom-up processing)
        items = sorted(fp_tree.item_counts.items(), key=lambda x: x[1])
        
        for item, support in items:
            # Create new frequent itemset
            new_itemset = alpha + [item]
            patterns[frozenset(new_itemset)] = support
            
            # Get conditional pattern base
            conditional_patterns = fp_tree.get_paths_for_item(item)
            
            if conditional_patterns:
                # Build conditional FP-tree
                conditional_transactions = []
                conditional_item_counts = Counter()
                
                for pattern, count in conditional_patterns:
                    conditional_transactions.extend([pattern] * count)
                    for pattern_item in pattern:
                        conditional_item_counts[pattern_item] += count
                
                # Filter frequent items in conditional pattern base
                conditional_frequent_items = {item: count for item, count in conditional_item_counts.items() 
                                            if count >= self.threshold}
                
                if conditional_frequent_items:
                    # Build conditional FP-tree
                    conditional_fp_tree = self._build_fp_tree(conditional_transactions, conditional_frequent_items)
                    
                    # Recursively mine patterns
                    conditional_patterns = self._mine_patterns(conditional_fp_tree, new_itemset)
                    patterns.update(conditional_patterns)
        
        return patterns
    
    def run_fpgrowth(self) -> Dict[int, Dict[frozenset, int]]:
        """Run the FP-Growth algorithm."""
        if self.threshold is None:
            self.set_threshold()
        
        print("\n" + "="*50)
        print("RUNNING FP-GROWTH ALGORITHM")
        print("="*50)
        
        start_time = perf_counter()
        
        # Step 1: Get frequent 1-itemsets
        print("Step 1: Finding frequent items...")
        frequent_items = self._get_frequent_items()
        print(f"Found {len(frequent_items)} frequent items")
        
        # Store frequent 1-itemsets
        frequent_1_itemsets = {frozenset([item]): count for item, count in frequent_items.items()}
        all_frequent_itemsets = {1: frequent_1_itemsets}
        
        # Step 2: Build FP-tree
        print("Step 2: Building FP-tree...")
        fp_tree = self._build_fp_tree(self.transactions, frequent_items)
        
        # Step 3: Mine frequent patterns
        print("Step 3: Mining frequent patterns...")
        all_patterns = self._mine_patterns(fp_tree)
        
        # Organize patterns by length
        for itemset, support in all_patterns.items():
            length = len(itemset)
            if length not in all_frequent_itemsets:
                all_frequent_itemsets[length] = {}
            all_frequent_itemsets[length][itemset] = support
        
        end_time = perf_counter()
        self.total_runtime = end_time - start_time
        
        # Print results
        for level in sorted(all_frequent_itemsets.keys()):
            itemsets = all_frequent_itemsets[level]
            print(f"\nLevel {level}: Found {len(itemsets)} frequent {level}-itemsets")
            self._print_itemsets(itemsets, level)
        
        print(f"\nTotal FP-Growth Runtime: {self.total_runtime:.4f} seconds")
        
        self.frequent_itemsets = all_frequent_itemsets
        return all_frequent_itemsets
    
    def _print_itemsets(self, itemsets: Dict[frozenset, int], level: int):
        """Print itemsets in a readable format."""
        if not itemsets:
            return
            
        self.stream_output(f"\nFrequent {level}-itemsets:")
        self.stream_output("-" * 40)
        
        # Sort by support count (descending)
        sorted_itemsets = sorted(itemsets.items(), key=lambda x: x[1], reverse=True)
        
        for itemset, count in sorted_itemsets:
            items_str = ', '.join(sorted(list(itemset)))
            support_pct = (count / self.transaction_count) * 100
            self.stream_output(f"  {{{items_str}}} - Support: {count} ({support_pct:.2f}%)")
    
    def get_summary(self) -> Dict:
        """Get a summary of the FP-Growth results."""
        if not self.frequent_itemsets:
            return {"error": "FP-Growth algorithm has not been run yet"}
        
        summary = {
            "total_transactions": self.transaction_count,
            "min_support_threshold": self.threshold,
            "min_support_percentage": self.min_support * 100,
            "levels_found": len(self.frequent_itemsets),
            "total_frequent_itemsets": sum(len(itemsets) for itemsets in self.frequent_itemsets.values()),
            "total_fpgrowth_runtime": f"{self.total_runtime:.4f} seconds"
        }
        
        for level, itemsets in self.frequent_itemsets.items():
            summary[f"level_{level}_count"] = len(itemsets)
        
        return summary
    
    def get_itemsets_dataframe(self) -> pd.DataFrame:
        """Convert frequent itemsets to a pandas DataFrame."""
        if not self.frequent_itemsets:
            return pd.DataFrame()
        
        rows = []
        for level, itemsets in self.frequent_itemsets.items():
            for itemset, count in itemsets.items():
                rows.append({
                    'Level': level,
                    'Itemset': ', '.join(sorted(list(itemset))),
                    'Items': list(itemset),
                    'Support_Count': count,
                    'Support_Percentage': (count / self.transaction_count) * 100
                })
        
        return pd.DataFrame(rows)
    
    def generate_association_rules(self, min_confidence: float = 0.5) -> pd.DataFrame:
        """Generate association rules from frequent itemsets.
        
        Args:
            min_confidence (float): Minimum confidence threshold (0-1)
            
        Returns:
            pd.DataFrame: Association rules with metrics
        """
        if not self.frequent_itemsets:
            print("Please run FP-Growth algorithm first")
            return pd.DataFrame()
        
        rules = []
        
        # Generate rules from itemsets of size 2 and above
        for level in range(2, len(self.frequent_itemsets) + 1):
            if level not in self.frequent_itemsets:
                continue
                
            for itemset, itemset_support in self.frequent_itemsets[level].items():
                items = list(itemset)
                
                # Generate all possible antecedent/consequent combinations
                from itertools import combinations
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent_set = frozenset(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        # Find support of antecedent
                        antecedent_support = self._find_itemset_support(antecedent_set)
                        
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= min_confidence:
                                consequent_support = self._find_itemset_support(consequent_set)
                                lift = confidence / (consequent_support / self.transaction_count)
                                
                                rules.append({
                                    'Antecedent': ', '.join(sorted(antecedent)),
                                    'Consequent': ', '.join(sorted(consequent_set)),
                                    'Support': itemset_support / self.transaction_count,
                                    'Confidence': confidence,
                                    'Lift': lift
                                })
        
        return pd.DataFrame(rules).sort_values('Confidence', ascending=False)
    
    def _find_itemset_support(self, itemset: frozenset) -> int:
        """Find support count for a given itemset."""
        for level_itemsets in self.frequent_itemsets.values():
            if itemset in level_itemsets:
                return level_itemsets[itemset]
        return 0

    def stream_output(self, text: str):
        """Stream output to file."""
        with open("result/fpgrowth_output.txt", "a") as f:
            f.write(f"{text}\n")

    def output(self):
        """Write complete output to file."""
        with open("result/fpgrowth_output.txt", "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write("ALGORITHM SUMMARY\n")
            f.write("="*50 + "\n")
            summary = self.get_summary()
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            # Generate association rules
            f.write("\n" + "="*50 + "\n")
            f.write("ASSOCIATION RULES\n")
            f.write("="*50 + "\n")
            rules_df = self.generate_association_rules(min_confidence=0.3)
            if not rules_df.empty:
                f.write(rules_df.head(10).to_string(index=False) + "\n")
            else:
                f.write("No association rules found with the given confidence threshold\n")

if __name__ == '__main__':
    try:
        fpgrowth = FPGrowthProcessor('data/e-commerce.csv', 
                                   unique_column='TransactionNo',
                                   group_column='ProductName',
                                   # sample=0.1  # Using 10% sample
                                   )  
        
        fpgrowth.set_threshold(0.01)  # 1% minimum support
        
        # Alternative test data
        # fpgrowth = FPGrowthProcessor('data/test.csv', 
        #                            unique_column='Transaction',
        #                            group_column='Item_name')
        # fpgrowth.set_threshold(0.15)
        
        results = fpgrowth.run_fpgrowth()
        
        fpgrowth.output()
        
        # Print summary
        print("\n" + "="*50)
        print("ALGORITHM SUMMARY")
        print("="*50)
        summary = fpgrowth.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Generate association rules
        print("\n" + "="*50)
        print("ASSOCIATION RULES")
        print("="*50)
        rules_df = fpgrowth.generate_association_rules(min_confidence=0.3)
        if not rules_df.empty:
            print(rules_df.head(10).to_string(index=False))
        else:
            print("No association rules found with the given confidence threshold")
            
    except FileNotFoundError:
        print("Data file not found.")