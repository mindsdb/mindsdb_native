import os
from pathlib import Path
import pickle
import shutil
import zipfile
import traceback
import uuid
import tempfile
from copy import deepcopy

from mindsdb_native.config import CONFIG
from mindsdb_native.__about__ import __version__
from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.libs.controllers.transaction import AnalyseTransaction
from mindsdb_native.libs.controllers.predictor import _get_memory_optimizations, _prepare_sample_settings, Predictor
from mindsdb_native.libs.helpers.multi_data_source import getDS
from mindsdb_native.libs.helpers.general_helpers import load_lmd, load_hmd
from mindsdb_native.libs.constants.mindsdb import (
    MODEL_STATUS_TRAINED,
    MODEL_STATUS_ERROR,
    TRANSACTION_ANALYSE
)
from mindsdb_native.libs.helpers.locking import MDBLock


def validate(to_predict, from_data, accuracy_score_functions, learn_args=None, test_args=None):
            if learn_args is None: learn_args = {}
            if test_args is None: test_args = {}

            name = str(uuid.uuid4()).replace('-','')
            predictor = Predictor(name)

            predictor.learn(to_predict, from_data, **learn_args)
            validation_data = predictor.transaction.input_data.validation_df

            accuracy = predictor.test(when_data=validation_data, accuracy_score_functions=accuracy_score_functions, **test_args)

            delete_model(name)

            return accuracy


def cross_validate(to_predict, from_data, accuracy_score_functions, k=5, learn_args=None, test_args=None):
    '''
        Probably required a change to generate a split into `k` folds, then manually setting those folds as train/test/predict.

        Would be problematic for timeseries data though.

        Alternatively we can just add train/test/valid split as advanced args and forgo the data splitter if they are specified (maybe have them be indexes so that the rest of the phases up to the splitter and run normally) + Do the splitting inside this function.

        Same problem with timeseries argument support though.
    '''
    raise NotImplementedError('Cross validation is not implemented yet')


def analyse_dataset(from_data, sample_settings=None):
    """
    Analyse the particular dataset being given
    """

    from_ds = getDS(from_data)
    transaction_type = TRANSACTION_ANALYSE

    sample_for_analysis, _ = _get_memory_optimizations(from_ds.df)

    sample_settings, sample_function = _prepare_sample_settings(
        sample_settings,
        sample_for_analysis
    )

    heavy_transaction_metadata = dict(
        name=None,
        from_data=from_ds,
        sample_function=sample_function,
        breakpoint = None
    )

    light_transaction_metadata = dict(
        version = str(__version__),
        name = None,
        model_columns_map = from_ds._col_map,
        type = transaction_type,
        sample_settings = sample_settings,
        tss={'is_timeseries':False},
        columns_to_ignore = [],
        data_preparation = {},
        predict_columns = [],
        empty_columns = [],
        force_column_usage = [],
        force_categorical_encoding = [],
        data_types = {},
        data_subtypes = {}
    )

    tx = AnalyseTransaction(
        session=None,
        light_transaction_metadata=light_transaction_metadata,
        heavy_transaction_metadata=heavy_transaction_metadata,
        logger=log
    )

    tx.run()

    return get_model_data(lmd=tx.lmd)


def export_storage(mindsdb_storage_dir='mindsdb_storage'):
    """
    If you want to export this mindsdb's instance storage to a file

    :param mindsdb_storage_dir: this is the full_path where you want to store a mind to, it will be a zip file
    :return: bool (True/False) True if mind was exported successfully
    """
    shutil.make_archive(
        base_name=mindsdb_storage_dir,
        format='zip',
        root_dir=CONFIG.MINDSDB_STORAGE_PATH
    )

    print(f'Exported mindsdb storage to {mindsdb_storage_dir}.zip')


def export_predictor(model_name):
    """Exports a Predictor to a zip file in the CONFIG.MINDSDB_STORAGE_PATH directory.

    :param model: a Predictor
    :param model_name: this is the name of the model you wish to export (defaults to the name of the passed Predictor)
    """
    with MDBLock('shared', 'predict_' + model_name):
        storage_file = model_name + '.zip'
        with zipfile.ZipFile(storage_file, 'w') as zip_fp:
            for file_name in os.listdir(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name)):
                full_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name, file_name)
                zip_fp.write(full_path, os.path.basename(full_path))

        print(f'Exported model to {storage_file}')


def rename_model(old_model_name, new_model_name):
    """
    If you want to rename an exported model.

    :param old_model_name: this is the name of the model you wish to rename
    :param new_model_name: this is the new name of the model
    :return: bool (True/False) True if predictor was renamed successfully
    """
    lock1 = MDBLock('exclusive', 'delete_' + new_model_name)
    lock2 = MDBLock('exclusive', 'delete_' + old_model_name)
    with lock1, lock2:

        if old_model_name == new_model_name:
            return True

        try:
            shutil.copy(
                os.path.join(CONFIG.MINDSDB_STORAGE_PATH, old_model_name),
                os.path.join(CONFIG.MINDSDB_STORAGE_PATH, new_model_name)
            )
        except Exception:
            return False

        lmd = load_lmd(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, old_model_name, 'light_model_metadata.pickle'))
        hmd = load_lmd(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, old_model_name, 'heavy_model_metadata.pickle'))

        lmd['name'] = new_model_name
        hmd['name'] = new_model_name

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            new_model_name, 'light_model_metadata.pickle'),
                'wb') as fp:
            pickle.dump(lmd, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            new_model_name, 'heavy_model_metadata.pickle'),
                'wb') as fp:
            pickle.dump(hmd, fp, protocol=pickle.HIGHEST_PROTOCOL)

        shutil.rmtree(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            old_model_name))
        return True


def delete_model(model_name):
    """
    If you want to delete exported model files.

    :param model_name: name of the model
    :return: bool (True/False) True if model was deleted
    """
    p = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name)
    if os.path.isdir(p):
        with MDBLock('exclusive', 'delete_' + model_name):
            shutil.rmtree(p)


def import_model(model_archive_path, new_name=None):
    """
    Import a mindsdb instance storage from an archive file.

    :param mindsdb_storage_dir: full_path that contains your mindsdb predictor zip file
    """
    previous_models = [str(p.name) for p in Path(CONFIG.MINDSDB_STORAGE_PATH).iterdir() if p.is_dir()]
    extract_dir = tempfile.mkdtemp(dir=CONFIG.MINDSDB_STORAGE_PATH)
    shutil.unpack_archive(model_archive_path, extract_dir=extract_dir)

    try:
        lmd = load_lmd(os.path.join(extract_dir, 'light_model_metadata.pickle'))
    except Exception:
        shutil.rmtree(extract_dir)
        raise

    if new_name is not None:
        lmd['name'] = new_name
    elif lmd['name'] is None:
        lmd['name'] = extract_dir

    if lmd['name'] in previous_models:
        shutil.rmtree(extract_dir)
        raise Exception(f"Model with name '{lmd['name']}' already exists.")

    shutil.move(
        extract_dir,
        os.path.join(CONFIG.MINDSDB_STORAGE_PATH, lmd['name'])
    )

    with MDBLock('exclusive', 'detele_' + lmd['name']):
        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, lmd['name'], 'light_model_metadata.pickle'), 'wb') as fp:
            pickle.dump(lmd, fp,protocol=pickle.HIGHEST_PROTOCOL)

    print('Model files loaded')
    return lmd['name']


def get_model_data(model_name=None, lmd=None):
    if model_name is None and lmd is None:
        raise ValueError('provide either model name or lmd')

    if lmd is not None:
        pass
    elif model_name is not None:
        with MDBLock('shared', 'get_data_' + model_name):
            lmd = load_lmd(os.path.join(
                CONFIG.MINDSDB_STORAGE_PATH,
                model_name,
                'light_model_metadata.pickle'
            ))

    # ADAPTOR CODE
    amd = {}

    amd['data_source'] = lmd.get('data_source_name')
    amd['useable_input_columns'] = lmd.get('useable_input_columns', [])

    amd['timeseries'] = {'is_timeseries': False}
    if 'tss' in lmd:
        if lmd['tss']['is_timeseries']:
            amd['timeseries']['is_timeseries'] = True
            amd['timeseries'] = {}
            amd['timeseries']['user_settings'] = lmd['tss']


    amd['data_analysis_v2'] = deepcopy(lmd.get('stats_v2'))
    # Remove keys that arent relevant to the GUI and JSON serializable:
    for target in amd['data_analysis_v2']:
        if 'train_std_dev' in amd['data_analysis_v2'][target]:
            del amd['data_analysis_v2'][target]['train_std_dev']
    # @TODO: Remove in the future

    amd['setup_args'] = lmd.get('setup_args')
    amd['test_data_plot'] = lmd.get('test_data_plot')
    amd['columns_to_ignore'] = lmd.get('columns_to_ignore')
    amd['columns'] = lmd.get('columns')
    amd['predict_columns'] = lmd.get('predict_columns')
    amd['output_class_distribution'] = lmd.get('output_class_distribution')

    if lmd['current_phase'] == MODEL_STATUS_TRAINED:
        amd['status'] = 'complete'
    elif lmd['current_phase'] == MODEL_STATUS_ERROR:
        amd['status'] = 'error'
        amd['stack_trace_on_error'] = lmd['stack_trace_on_error']
        amd['error_explanation'] = lmd['error_explanation']
    else:
        amd['status'] = 'training'

    # Shared keys
    for k in ['name', 'version', 'is_active', 'predict', 'current_phase',
    'train_end_at', 'updated_at', 'created_at','data_preparation', 'validation_set_accuracy', 'report_uuid']:
        if k == 'predict':
            amd[k] = lmd['predict_columns']
        elif k in lmd:
            amd[k] = lmd[k]
            if k == 'validation_set_accuracy':
                if lmd['validation_set_accuracy'] is not None:
                    amd['accuracy'] = round(lmd['validation_set_accuracy'], 3)
                else:
                    amd['accuracy'] = None
        else:
            amd[k] = None

    if 'validation_set_accuracy_r2' in lmd:
        amd['accuracy_r2'] = lmd['validation_set_accuracy_r2']

    amd['model_analysis'] = []

    for col in lmd['model_columns_map'].keys():
        if col in lmd['columns_to_ignore']:
            continue

        amd['force_vectors'] = {}
        if col in lmd['predict_columns']:
            if 'confusion_matrices' in lmd and col in lmd['confusion_matrices']:
                confusion_matrix = lmd['confusion_matrices'][col]
            else:
                confusion_matrix = None

            if 'accuracy_samples' in lmd and col in lmd['accuracy_samples']:
                accuracy_samples = lmd['accuracy_samples'][col]
            else:
                accuracy_samples = None

            # Model analysis building for each of the predict columns
            mao = {
                'column_name': col
                ,'overall_input_importance': {
                    "type": "categorical"
                    ,"x": []
                    ,"y": []
                }
              ,"train_accuracy_over_time": {
                "type": "categorical",
                "x": [],
                "y": []
              }
              ,"test_accuracy_over_time": {
                "type": "categorical",
                "x": [],
                "y": []
              }
              ,"accuracy_histogram": {
                    "x": []
                    ,"y": []
                    ,'x_explained': []
              }
              ,"confusion_matrix": confusion_matrix
              ,"accuracy_samples": accuracy_samples
            }


            # This is a check to see if model analysis has run on this data
            if 'model_accuracy' in lmd and lmd['model_accuracy'] is not None and 'train' in lmd['model_accuracy'] and 'combined' in lmd['model_accuracy']['train'] and lmd['model_accuracy']['train']['combined'] is not None:
                train_acc = lmd['model_accuracy']['train']['combined']
                test_acc = lmd['model_accuracy']['test']['combined']

                for i in range(0,len(train_acc)):
                    mao['train_accuracy_over_time']['x'].append(i)
                    mao['train_accuracy_over_time']['y'].append(train_acc[i])

                for i in range(0,len(test_acc)):
                    mao['test_accuracy_over_time']['x'].append(i)
                    mao['test_accuracy_over_time']['y'].append([i])

            if 'model_accuracy' in lmd and lmd['model_accuracy'] is not None and lmd['column_importances'] is not None:
                mao['accuracy_histogram']['x'] = [f'{x}' for x in lmd['accuracy_histogram'][col]['buckets']]
                mao['accuracy_histogram']['y'] = lmd['accuracy_histogram'][col]['accuracies']

                for icol in lmd['model_columns_map'].keys():
                    if icol in lmd['columns_to_ignore']:
                        continue
                    if icol not in lmd['predict_columns']:
                        try:
                            mao['overall_input_importance']['x'].append(icol)
                            mao['overall_input_importance']['y'].append(round(lmd['column_importances'][icol],1))
                        except Exception:
                            print(f'No column importances found for {icol} !')

            for key in ['train_data_accuracy', 'test_data_accuracy', 'valid_data_accuracy']:
                if key in lmd:
                    mao[key] = lmd[key]

            amd['model_analysis'].append(mao)

    return amd
