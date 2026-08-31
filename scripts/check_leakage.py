import pandas as pd
import random

random.seed(42)
out_dir = 'data/run1'
qrels = pd.read_parquet(out_dir + '/qrels.parquet')
train_q = pd.read_parquet(out_dir + '/splits/train.parquet')
dev_q = pd.read_parquet(out_dir + '/splits/dev.parquet')
test_q = pd.read_parquet(out_dir + '/splits/test.parquet')

sample_ids = random.sample(list(train_q['query_id']), 5)
print('Sampled Query IDs:', sample_ids)

dev_p = set(qrels[qrels['query_id'].isin(dev_q['query_id'])]['product_id'])
test_p = set(qrels[qrels['query_id'].isin(test_q['query_id'])]['product_id'])

for q in sample_ids:
    cands = set(qrels[qrels['query_id'] == q]['product_id'])
    print(f'Query {q} candidates leaked to dev: {len(cands.intersection(dev_p))} / {len(cands)}')
    print(f'Query {q} candidates leaked to test: {len(cands.intersection(test_p))} / {len(cands)}')
