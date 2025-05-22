from pathlib import Path

import kaggle

def download_dataset(source: str, rename: str = None):
    """ Downloads a dataset from Kaggle using the given source identifier.

    Args:
        source (str): The Kaggle dataset identifier `(e.g., 'user/dataset-name').`
        rename (str, optional): New filename to rename the downloaded CSV file. 
                                If not provided, the original filename is kept.
    """
    
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(source, path='data/', unzip=True)
    
    # Rename file
    if rename != None:
        for file in Path('data').glob('*.csv'):
            if file.name != 'e-commerce.csv':
                file.rename('data/e-commerce.csv')
                break

    
if __name__ == '__main__':
    download_dataset('gabrielramos87/an-online-shop-business', 
                     'e-commerce.csv')