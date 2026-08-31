import pandas as pd
from sqlalchemy import create_engine
import traceback
import os
from dotenv import load_dotenv

load_dotenv()
db_url = os.environ.get('DATABASE_URL').replace('postgresql://', 'postgresql+psycopg2://')
engine = create_engine(db_url)
products = pd.read_parquet('data/run1/products.parquet')
try:
    products.to_sql('products', engine, if_exists='append', index=False, method='multi', chunksize=1000)
except Exception as e:
    with open('error.txt', 'w', encoding='utf-8') as f:
        f.write(traceback.format_exc())
