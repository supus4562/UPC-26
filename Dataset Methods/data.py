"""Fixed, fingerprinted train/test partitions; no implicit splitting at training time."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from physics_predictors import OBSERVATIONS, CROSSING


def digest(table):
    names = ['id'] + OBSERVATIONS + CROSSING
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode())
        h.update(np.asarray(table[name].to_numpy(), dtype='<i8' if name == 'id' else '<f8').tobytes())
    return h.hexdigest()


def read_table(path):
    path = Path(path)
    files = sorted(path.glob('part-*.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'No part-*.parquet in {path}')
    table = pa.concat_tables([pq.read_table(f) for f in files])
    if 'id' not in table.column_names:
        table = table.append_column('id', pa.array(np.arange(len(table), dtype=np.int64)))
    if len(np.unique(table['id'].to_numpy())) != len(table):
        raise ValueError('Duplicate kick IDs')
    return table.sort_by([('id', 'ascending')])


def create_split(source, output, seed=42):
    source, output = Path(source), Path(output)
    table = read_table(source)
    if len(table) < 5:
        raise ValueError('At least five accepted kicks needed for an 80/20 split')
    fingerprint = digest(table)
    config = dict(version=1, source_sha256=fingerprint, seed=seed, train_fraction=.8)
    manifest = output/'split.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old[k] != v for k, v in config.items()):
            raise ValueError('Split already exists with different data or seed; use a fresh destination')
        load_partition(output, 'train'); load_partition(output, 'test')
        return old
    if any((output/s).exists() for s in ('train','test')):
        raise ValueError('Incomplete split found; use a fresh destination')
    order = np.random.default_rng(seed).permutation(len(table))
    ntrain = int(.8*len(table))
    for name, indices in [('train',order[:ntrain]), ('test',order[ntrain:])]:
        part = table.take(pa.array(np.sort(indices)))
        folder = output/name
        folder.mkdir(parents=True, exist_ok=False)
        pq.write_table(part, folder/'part-00000.parquet', compression='zstd')
        config[name+'_rows'] = len(part)
        config[name+'_sha256'] = digest(part)
    manifest.write_text(json.dumps(config, indent=2))
    return config


def load_partition(root, split):
    root = Path(root)
    meta = json.loads((root/'split.json').read_text())
    table = read_table(root/split)
    if digest(table) != meta[split+'_sha256']:
        raise ValueError(f'{split} partition fingerprint mismatch')
    matrix = np.column_stack([table[n].to_numpy() for n in OBSERVATIONS+CROSSING])
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:,7],11,atol=1e-8,rtol=0):
        raise ValueError('Expected finite Team 358 coordinates, y_goal=11')
    return matrix[:,:6], matrix[:,[6,8]], table['id'].to_numpy(), meta


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description='Split existing accepted kicks without resimulation')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    print(json.dumps(create_split(a.source,a.output,a.seed), indent=2))
