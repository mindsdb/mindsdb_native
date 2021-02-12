import os
from io import BytesIO, StringIO
import csv
import codecs
import json
import traceback

import pandas as pd
import requests

from mindsdb_native.libs.data_types.data_source import DataSource
from mindsdb_native.libs.data_types.mindsdb_logger import log


def clean_row(row):
    n_row = []
    for cell in row:
        if str(cell) in ['', ' ', '  ', 'NaN', 'nan', 'NA']:
            n_row.append(None)
        else:
            n_row.append(cell)

    return n_row


class FileDS(DataSource):
    def __init__(self, file, clean_rows=True, custom_parser=None):
        """
        Setup from file
        :param file: fielpath or url
        :param clean_rows: if you want to clean rows for strange null values
        :param custom_parser: if you want to parse the file with some custom parser
        """
        if not isinstance(file, str):
            raise ValueError("'file' must be string")
        super().__init__()
        self.file = file
        self.clean_rows = clean_rows
        self.custom_parser = custom_parser
        self.dialect = None

    def _handle_source(self):
        self._file_name = os.path.basename(self.file)

        # get file data io, format and dialect
        data, fmt, self.dialect = self._getDataIo(self.file)
        data.seek(0)  # make sure we are at 0 in file pointer

        if self.custom_parser:
            header, file_data = self.custom_parser(data, fmt)

        elif fmt == 'csv':
            csv_reader = list(csv.reader(data, self.dialect))
            header = csv_reader[0]
            file_data =  csv_reader[1:]

        elif fmt in ['xlsx', 'xls']:
            data.seek(0)
            df = pd.read_excel(data)
            header = df.columns.values.tolist()
            file_data = df.values.tolist()

        elif fmt == 'json':
            data.seek(0)
            json_doc = json.loads(data.read())
            df = pd.json_normalize(json_doc, max_level=0)
            header = df.columns.values.tolist()
            file_data = df.values.tolist()

        else:
            raise ValueError('Could not load file into any format, supported formats are csv, json, xls, xlsx')

        if self.clean_rows:
            file_list_data = [clean_row(row) for row in file_data]
        else:
            file_list_data = file_data

        col_map = dict((col, col) for col in header)
        return pd.DataFrame(file_list_data, columns=header), col_map

    def query(self, q=None):
        try:
            return self._handle_source()
        except Exception as e:
            log.error(f"Error creating dataframe from handled data: {e}")
            log.error("pd.read_csv data handler would be used.")
            df = pd.read_csv(self.file, sep=self.dialect.delimiter)
            col_map = dict((col, col) for col in df.columns)
            return df, col_map

    def _getDataIo(self, file):
        """
        This gets a file either url or local file and defiens what the format is as well as dialect
        :param file: file path or url
        :return: data_io, format, dialect
        """

        ############
        # get file as io object
        ############

        data = BytesIO()

        # get data from either url or file load in memory
        if file.startswith('http:') or file.startswith('https:'):
            print('Getting file')
            try:
                print(file)
                r = requests.get(file, stream=True)
            except Exception as e:
                print(e)
            print('Got file')
            if r.status_code == 200:
                for chunk in r:
                    data.write(chunk)
            data.seek(0)
            print('Read data')

        # else read file from local file system
        else:
            try:
                with open(file, 'rb') as fp:
                    data = BytesIO(fp.read())
            except Exception as e:
                error = 'Could not load file, possible exception : {exception}'.format(exception = e)
                log.error(error)
                raise ValueError(error)


        dialect = None

        ############
        # check for file type
        ############

        # try to guess if its an excel file
        xlsx_sig = b'\x50\x4B\x05\06'
        xlsx_sig2 = b'\x50\x4B\x03\x04'
        xls_sig = b'\x09\x08\x10\x00\x00\x06\x05\x00'

        # different whence, offset, size for different types
        excel_meta = [ ('xls', 0, 512, 8), ('xlsx', 2, -22, 4)]

        for filename, whence, offset, size in excel_meta:

            try:
                data.seek(offset, whence)  # Seek to the offset.
                bytes = data.read(size)  # Capture the specified number of bytes.
                data.seek(0)
                codecs.getencoder('hex')(bytes)

                if bytes == xls_sig:
                    return data, 'xls', dialect
                elif bytes == xlsx_sig:
                    return data, 'xlsx', dialect

            except Exception:
                data.seek(0)

        # if not excel it can be a json file or a CSV, convert from binary to stringio

        byte_str = data.read()
        # Move it to StringIO
        try:
            # Handle Microsoft's BOM "special" UTF-8 encoding
            if byte_str.startswith(codecs.BOM_UTF8):
                data = StringIO(byte_str.decode('utf-8-sig'))
            else:
                data = StringIO(byte_str.decode('utf-8'))

        except Exception:
            log.error(traceback.format_exc())
            log.error('Could not load into string')

        # see if its JSON
        buffer = data.read(100)
        data.seek(0)
        text = buffer.strip()
        # analyze first n characters
        if len(text) > 0:
            text = text.strip()
            # it it looks like a json, then try to parse it
            if text.startswith('{') or text.startswith('['):
                try:
                    json.loads(data.read())
                    data.seek(0)
                    return data, 'json', dialect
                except Exception:
                    data.seek(0)
                    return data, None, dialect

        # lets try to figure out if its a csv
        try:
            data.seek(0)
            first_few_lines = []
            i = 0

            # need to have sample to deduce a dialect
            # but it is not a good idea to deduce dialect by header row
            # data[0]
            for i, line in enumerate(data):
                first_few_lines.append(line)
                if i > 10:
                    break

            accepted_delimiters = [',','\t', ';']

            #provide sample from data if it is possible
            dialect = csv.Sniffer().sniff(''.join(first_few_lines), delimiters=accepted_delimiters)
            data.seek(0)
            # if csv dialect identified then return csv
            if dialect:
                return data, 'csv', dialect
            return data, None, dialect
        except Exception:
            data.seek(0)
            log.error('Could not detect format for this file')
            log.error(traceback.format_exc())
            # No file type identified
            return data, None, dialect

    def name(self):
        return 'File, {}'.format(os.path.basename(self.file))
