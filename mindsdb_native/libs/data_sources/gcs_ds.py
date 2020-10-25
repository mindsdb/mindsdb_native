import os 

from google.cloud import storage

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_sources.file_ds import FileDS
from mindsdb_native import F


class GCSDS(DataSource):

    def _setup(self, bucket_name, file_path, path_to_auth_json):

        self._bucket_name = bucket_name
        self._file_name = os.path.basename(file_path)

        gc_client = storage.Client.from_service_account_json(json_credentials_path=path_to_auth_json)
        bucket = gc_client.get_bucket(bucket_name)

        blob = storage.Blob(file_path, bucket)

        self.tmp_file_name = '.tmp_mindsdb_data_file'

        with open(self.tmp_file_name, 'wb') as fw:
            gc_client.download_blob_to_file(blob, fw)

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