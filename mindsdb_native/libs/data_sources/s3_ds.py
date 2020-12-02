import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_sources.file_ds import FileDS
from mindsdb_native import F


class S3DS(DataSource):
    def __init__(self, bucket_name, file_path, access_key=None,
                 secret_key=None, use_default_credentails=False):
        super().__init__()
        self.bucket_name = bucket_name
        self.file_path = file_path
        self.access_key = access_key
        self.secret_key = secret_key
        self.use_default_credentails = use_default_credentails

    def query(self, q):
        if self.access_key is not None and secret_key is not None:
            s3 = boto3.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key)
        elif use_default_credentails:
            s3 = boto3.client('s3')
        else:
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        with open(self.tmp_file_name, 'wb') as fw:
            s3.download_fileobj(self.bucket_name, self.file_path, fw)

        tmp_file_name = '.tmp_mindsdb_data_file'

        file_ds = FileDS(tmp_file_name)

        os.remove(tmp_file_name)

        return file_ds.df, file_ds._col_map

    def name(self):
        return 'S3 - {}/{}'.format(self.bucket_name, os.path.basename(self.file_path))
