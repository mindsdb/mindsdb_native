import os

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_sources.file_ds import FileDS
from mindsdb_native import F


class S3DS(DataSource):

    def _setup(self, bucket_name, file_path, access_key=None,
               secret_key=None,use_default_credentails=False):

        self._bucket_name = bucket_name
        self._file_name = os.path.basename(file_path)

        if access_key is not None and secret_key is not None:
            s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        elif use_default_credentails:
            s3 = boto3.client('s3')
        else:
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        self.tmp_file_name = '.tmp_mindsdb_data_file'

        with open(self.tmp_file_name, 'wb') as fw:
            s3.download_fileobj(bucket_name, file_path, fw)

        file_ds = FileDS(self.tmp_file_name)
        return file_ds._df, file_ds._col_map

    def _cleanup(self):
        os.remove(self.tmp_file_name)


    def name(self):
        return '{}: {}/{}'.format(
            self.__class__.__name__,
            self._bucket_name,
            self._file_name
        )