import pandas as pd
from itertools import combinations
from typing import List, Dict, Optional, DefaultDict, FrozenSet
from collections import defaultdict
from time import perf_counter

class AprioriProcessor:
    """An implementation of the Apriori algorithm for frequent itemset mining."""
    
    def __init__(self, path: str, unique_column: str, group_column: str, sample: Optional[float] = None):
        """Initialize the AprioriProcessor with a dataset.

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
        self.transactions = df_clean.groupby(self.unique_column)[self.group_column].apply(set).tolist()
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
        self.threshold : int = max(1, round(min_support * self.transaction_count))
        print(f'Minimum support threshold: {self.threshold} transactions ({min_support*100:.1f}%)')
    
    def _get_frequent_1_itemsets(self) -> Dict[frozenset, int]:
        """Generate frequent 1-itemsets."""
        item_counts: DefaultDict[str, int] = defaultdict(int)
        
        # Count occurrences of each item
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        frequent_1_itemsets: Dict[FrozenSet[str], int] = {}
        for item, count in item_counts.items():
            if count >= self.threshold:
                frequent_1_itemsets[frozenset([item])] = count
        
        return frequent_1_itemsets
    
    def _generate_candidates(self, frequent_itemsets: Dict[FrozenSet[str], int], k: int) -> List[frozenset]:
        """Generate candidate itemsets of size k from frequent itemsets of size k-1."""
        items = []
        for itemset in frequent_itemsets.keys():
            items.extend(list(itemset))
        
        # Get unique items and sort for consistent ordering
        unique_items = sorted(set(items))
        
        # Generate k-combinations
        candidates = []
        for combo in combinations(unique_items, k):
            candidate = frozenset(combo)
            
            # Apriori property: all (k-1)-subsets must be frequent
            if self._has_frequent_subsets(candidate, frequent_itemsets, k-1):
                candidates.append(candidate)
        
        return candidates
    
    def _has_frequent_subsets(self, candidate: frozenset, frequent_itemsets: Dict[frozenset, int], k: int) -> bool:
        """Check if all k-subsets of candidate are frequent."""
        if k == 1:
            return True
        
        # If an itemset is frequent, all of its subsets must also be frequent.
        for combo in combinations(candidate, k):
            subset = frozenset(combo)
            if subset not in frequent_itemsets:
                return False
        return True
    
    def _count_support(self, candidates: List[frozenset]) -> Dict[frozenset, int]:
        """Count support for candidate itemsets."""
        candidate_counts = {candidate: 0 for candidate in candidates}
        
        for transaction in self.transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1
        
        return candidate_counts
    
    def _filter_frequent(self, candidate_counts: Dict[frozenset, int]) -> Dict[frozenset, int]:
        """Filter candidates that meet minimum support threshold."""
        return {itemset: count for itemset, count in candidate_counts.items() 
                if count >= self.threshold}
    
    def run_apriori(self) -> Dict[int, Dict[frozenset, int]]:
        """Run the Apriori algorithm."""
        if self.threshold is None:
            self.set_threshold()
        
        print("\n" + "="*50)
        print("RUNNING APRIORI ALGORITHM")
        print("="*50)
        
        self.start_total = perf_counter()
        self.level_runtimes: Dict[int, float] = {}
    
        # Initialize with frequent 1-itemsets
        start_fi_1 = perf_counter()
        frequent_itemsets = self._get_frequent_1_itemsets()
        all_frequent_itemsets = {1: frequent_itemsets}
        
        result_fi_1 = perf_counter() - start_fi_1
        self.level_runtimes[1] = result_fi_1
        print(f"Level {1} runtime: {result_fi_1:.4f} seconds")
        
        
        print(f"\nLevel 1: Found {len(frequent_itemsets)} frequent 1-itemsets")
        self._print_itemsets(frequent_itemsets, 1)
        
        k = 2
        while frequent_itemsets:
            level_start = perf_counter()
            
            # Generate candidates
            candidates = self._generate_candidates(frequent_itemsets, k)
            
            if not candidates:
                break
            
            print(f"\nLevel {k}: Generated {len(candidates)} candidates")
            
            # Count support for candidates
            candidate_counts = self._count_support(candidates)
            
            # Filter frequent itemsets
            frequent_itemsets = self._filter_frequent(candidate_counts)
            
            level_duration = perf_counter() - level_start
            self.level_runtimes[k] = level_duration
            
            if frequent_itemsets:
                all_frequent_itemsets[k] = frequent_itemsets
                print(f"Level {k}: Found {len(frequent_itemsets)} frequent {k}-itemsets")
                print(f"Level {k} runtime: {level_duration:.4f} seconds")
                self._print_itemsets(frequent_itemsets, k)
            else:
                print(f"Level {k}: No frequent itemsets found")
                print(f"Level {k} runtime: {level_duration:.4f} seconds")
            
            k += 1
            
        self.result_total = perf_counter() - self.start_total
        print(f"\nTotal Apriori Runtime: {self.result_total:.4f} seconds")
    
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
            # print(f"  {{{items_str}}} - Support: {count} ({support_pct:.2f}%)")
            self.stream_output(f"  {{{items_str}}} - Support: {count} ({support_pct:.2f}%)")
    
    def get_summary(self) -> Dict:
        """Get a summary of the Apriori results."""
        if not self.frequent_itemsets:
            return {"error": "Apriori algorithm has not been run yet"}
        
        summary = {
            "total_transactions": self.transaction_count,
            "min_support_threshold": self.threshold,
            "min_support_percentage": self.min_support * 100,
            "levels_found": len(self.frequent_itemsets),
            "total_frequent_itemsets": sum(len(itemsets) for itemsets in self.frequent_itemsets.values())
        }
        
        for level, itemsets in self.frequent_itemsets.items():
            summary[f"level_{level}_count"] = f"{len(itemsets)}, in {self.level_runtimes[level]:.4f} seconds"
        
        summary["total_apriori_runtime"] = f"{self.result_total:.4f} seconds"
        
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
            print("Please run Apriori algorithm first")
            return pd.DataFrame()
        
        rules = []
        
        # Generate rules from itemsets of size 2 and above
        for level in range(2, len(self.frequent_itemsets) + 1):
            if level not in self.frequent_itemsets:
                continue
                
            for itemset, itemset_support in self.frequent_itemsets[level].items():
                items = list(itemset)
                
                # Generate all possible antecedent/consequent combinations
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent_set = frozenset(antecedent)
                        consequent_set = itemset - antecedent_set
                        
                        # Find support of antecedent
                        antecedent_support = self._find_itemset_support(antecedent_set)
                        
                        if antecedent_support > 0:
                            confidence = itemset_support / antecedent_support
                            
                            if confidence >= min_confidence:
                                lift = confidence / (self._find_itemset_support(consequent_set) / self.transaction_count)
                                
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

    def stream_output(self, text : str):
        with open("result/apriori_output.txt", "a") as f:
            f.write(f"{text}\n")

    def output(self):
        with open("result/apriori_output.txt", "a") as f:
            f.write("\n" + "="*50 + "\n")
            f.write("ALGORITHM SUMMARY\n")
            f.write("="*50 + "\n")
            summary = apriori.get_summary()
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            # Generate association rules
            f.write("\n" + "="*50 + "\n")
            f.write("ASSOCIATION RULES\n")
            f.write("="*50 + "\n")
            rules_df = apriori.generate_association_rules(min_confidence=0.3)
            if not rules_df.empty:
                f.write(rules_df.head(10).to_string(index=False) + "\n")
            else:
                f.write("No association rules found with the given confidence threshold\n")
    
    

if __name__ == '__main__':
    try:
        apriori = AprioriProcessor('data/e-commerce.csv', 
                                   unique_column='TransactionNo',
                                   group_column='ProductName',
                                #    sample=0.1 # Using 10% sample
                                )  
        
        apriori.set_threshold(0.01)  # 1% minimum support
        
        # apriori = AprioriProcessor('data/test.csv', 
        #                     unique_column='Transaction',
        #                     group_column='Item_name')
        
        # apriori.set_threshold(0.15)
        
        results = apriori.run_apriori()
        
        apriori.output()
        
        # Print summary
        print("\n" + "="*50)
        print("ALGORITHM SUMMARY")
        print("="*50)
        summary = apriori.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Generate association rules
        print("\n" + "="*50)
        print("ASSOCIATION RULES")
        print("="*50)
        rules_df = apriori.generate_association_rules(min_confidence=0.3)
        if not rules_df.empty:
            print(rules_df.head(10).to_string(index=False))
        else:
            print("No association rules found with the given confidence threshold")
            
    except FileNotFoundError:
        print("Data file not found.")