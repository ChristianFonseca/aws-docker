import pandas as pd
import boto3
from io import BytesIO, StringIO
from rimac_analytics_api.utils import exploration as raex

s3 = boto3.client('s3')


def get_bucket_and_key_from_full_path(full_path):
    if full_path.find('.') < 0:
        full_path += '/'
    bucket = full_path[:full_path.find('/')]
    prefix = full_path[full_path.find('/')+1:]
    return bucket, prefix


def get_s3_keys(full_path, get_all=True):
    """Get a list of all keys in an S3 bucket."""

    bucket, prefix = get_bucket_and_key_from_full_path(full_path)

    keys = []

    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        keys.extend([obj['Key'] for obj in resp['Contents']])
        if not get_all:
            break
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

    keys = [x.replace(prefix, '') for x in keys if x != prefix]
    types = list(set([x[x.rfind('.'):] for x in keys]))
    keys = {'keys': keys}

    for _type in types:
        keys[_type] = [key for key in keys['keys'] if key.endswith(_type)]

    return keys


def get_summary_s3(full_path, suffix='_summary.csv', **kwargs):
    bucket = full_path[:full_path.find('/')]
    path = full_path[full_path.find('/') + 1:]
    files = s3.list_objects(Bucket=bucket, Prefix=path)['Contents']
    files = [x['Key'] for x in files]
    files = [x for x in files if (x.endswith('.csv') or x.endswith('.csv.gz')) and not x.endswith(suffix)]
    print('Path:', full_path)
    lista = []

    for i, file in enumerate(files):
        obj = s3.get_object(Bucket=bucket, Key=file)
        compression = 'gzip' if file.endswith('.csv.gz') else 'infer'
        if len(files) == 1:
            df = pd.read_csv(BytesIO(obj['Body'].read()), compression=compression, dtype=str,
                             encoding='latin1')
        else:
            df0 = pd.read_csv(BytesIO(obj['Body'].read()), compression=compression, dtype=str,
                              encoding='latin1')
            lista.append(df0)
            print(i, end=', ')

    if len(files) > 1:
        df = pd.concat(lista)
        print('')

    del lista

    path_save = file.replace('.csv', '').replace('.gz', '') + suffix

    csv_buffer = StringIO()
    raex.get_descriptive(df, **kwargs).to_csv(csv_buffer)

    s3.put_object(Bucket=bucket, Body=csv_buffer.getvalue(), Key=path_save)
    print('¡¡¡ Saved !!!\n')

