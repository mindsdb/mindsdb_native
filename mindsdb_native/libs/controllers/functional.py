import os
import pickle
import shutil
import zipfile
import traceback
import uuid

from mindsdb_native.config import CONFIG
from mindsdb_native.__about__ import __version__
from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.libs.controllers.transaction import AnalyseTransaction
from mindsdb_native.libs.controllers.predictor import _get_memory_optimizations, _prepare_sample_settings, Predictor
from mindsdb_native.libs.helpers.multi_data_source import getDS
from mindsdb_native.libs.constants.mindsdb import (MODEL_STATUS_TRAINED,
                                                   MODEL_STATUS_ERROR,
                                                   TRANSACTION_ANALYSE)
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

    sample_for_analysis, sample_for_training, disable_lightwood_transform_cache = _get_memory_optimizations(
        from_ds.df)
    sample_settings, sample_function = _prepare_sample_settings(sample_settings,
                                                sample_for_analysis,
                                                sample_for_training)

    heavy_transaction_metadata = dict(
        name=None,
        from_data=from_ds,
        sample_function=sample_function
    )

    light_transaction_metadata = dict(
        version = str(__version__),
        name = None,
        model_columns_map = from_ds._col_map,
        type = transaction_type,
        sample_settings = sample_settings,
        model_is_time_series = False,
        model_group_by = [],
        model_order_by = [],
        columns_to_ignore = [],
        data_preparation = {},
        predict_columns = [],
        empty_columns = [],
        handle_foreign_keys = True,
        force_categorical_encoding = [],
        data_types = {},
        data_subtypes = {},
        breakpoint = None
    )

    tx = AnalyseTransaction(
        session=None,
        light_transaction_metadata=light_transaction_metadata,
        heavy_transaction_metadata=heavy_transaction_metadata,
        logger=log
    )

    md = get_model_data(lmd=tx.lmd)
    return md


def export_storage(mindsdb_storage_dir='mindsdb_storage'):
    """
    If you want to export this mindsdb's instance storage to a file

    :param mindsdb_storage_dir: this is the full_path where you want to store a mind to, it will be a zip file
    :return: bool (True/False) True if mind was exported successfully
    """
    shutil.make_archive(base_name=mindsdb_storage_dir, format='zip',
                        root_dir=CONFIG.MINDSDB_STORAGE_PATH)
    print(f'Exported mindsdb storage to {mindsdb_storage_dir}.zip')


def export_predictor(model_name):
    """Exports a Predictor to a zip file in the CONFIG.MINDSDB_STORAGE_PATH directory.

    :param model: a Predictor
    :param model_name: this is the name of the model you wish to export (defaults to the name of the passed Predictor)
    """
    with MDBLock('shared', 'predict_' + model_name):
        storage_file = model_name + '.zip'
        with zipfile.ZipFile(storage_file, 'w') as zip_fp:
            for file_name in [model_name + '_heavy_model_metadata.pickle',
                            model_name + '_light_model_metadata.pickle',
                            model_name + '_lightwood_data']:
                full_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, file_name)
                zip_fp.write(full_path, os.path.basename(full_path))

            # If the backend is ludwig, save the ludwig files
            try:
                ludwig_model_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                                                model_name + '_ludwig_data')
                for root, dirs, files in os.walk(ludwig_model_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        zip_fp.write(full_path,
                                    full_path[len(CONFIG.MINDSDB_STORAGE_PATH):])
            except Exception:
                pass

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

        moved_a_backend = False
        for extension in ['_lightwood_data', '_ludwig_data']:
            shutil.move(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                                    old_model_name + extension),
                        os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                                    new_model_name + extension))
            moved_a_backend = True

        if not moved_a_backend:
            return False

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            old_model_name + '_light_model_metadata.pickle'),
                'rb') as fp:
            lmd = pickle.load(fp)

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            old_model_name + '_heavy_model_metadata.pickle'),
                'rb') as fp:
            hmd = pickle.load(fp)

        lmd['name'] = new_model_name
        hmd['name'] = new_model_name

        renamed_one_backend = False
        try:
            lmd['ludwig_data']['ludwig_save_path'] = lmd['ludwig_data'][
                'ludwig_save_path'].replace(old_model_name, new_model_name)
            renamed_one_backend = True
        except Exception:
            pass

        try:
            lmd['lightwood_data']['save_path'] = lmd['lightwood_data'][
                'save_path'].replace(old_model_name, new_model_name)
            renamed_one_backend = True
        except Exception:
            pass

        if not renamed_one_backend:
            return False

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            new_model_name + '_light_model_metadata.pickle'),
                'wb') as fp:
            pickle.dump(lmd, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            new_model_name + '_heavy_model_metadata.pickle'),
                'wb') as fp:
            pickle.dump(hmd, fp, protocol=pickle.HIGHEST_PROTOCOL)

        os.remove(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            old_model_name + '_light_model_metadata.pickle'))
        os.remove(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,
                            old_model_name + '_heavy_model_metadata.pickle'))
        return True


def delete_model(model_name):
    """
    If you want to delete exported model files.

    :param model_name: name of the model
    :return: bool (True/False) True if model was deleted
    """
    with MDBLock('exclusive', 'delete_' + model_name):
        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name + '_light_model_metadata.pickle'), 'rb') as fp:
            lmd = pickle.load(fp)

            try:
                os.remove(lmd['lightwood_data']['save_path'])
            except Exception:
                pass

            try:
                shutil.rmtree(lmd['ludwig_data']['ludwig_save_path'])
            except Exception:
                pass

        for file_name in [model_name + '_heavy_model_metadata.pickle',
                        model_name + '_light_model_metadata.pickle']:
            os.remove(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, file_name))


def _import_model_by_name(model_name):
    with MDBLock('exclusive', 'detele_' + model_name):
        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name + '_light_model_metadata.pickle'), 'rb') as fp:
            lmd = pickle.load(fp)

        if 'ludwig_data' in lmd and 'ludwig_save_path' in lmd['ludwig_data']:
            lmd['ludwig_data']['ludwig_save_path'] = str(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,os.path.basename(lmd['ludwig_data']['ludwig_save_path'])))

        if 'lightwood_data' in lmd and 'save_path' in lmd['lightwood_data']:
            lmd['lightwood_data']['save_path'] = str(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,os.path.basename(lmd['lightwood_data']['save_path'])))

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name + '_light_model_metadata.pickle'), 'wb') as fp:
            pickle.dump(lmd, fp,protocol=pickle.HIGHEST_PROTOCOL)


def import_model(model_archive_path):
    """
    Import a mindsdb instance storage from an archive file.

    :param mindsdb_storage_dir: full_path that contains your mindsdb predictor zip file
    """
    previous_models = os.listdir(CONFIG.MINDSDB_STORAGE_PATH)
    shutil.unpack_archive(model_archive_path, extract_dir=CONFIG.MINDSDB_STORAGE_PATH)

    new_model_files = set(os.listdir(CONFIG.MINDSDB_STORAGE_PATH)) - set(previous_models)
    model_names = []
    for file in new_model_files:
        if '_light_model_metadata.pickle' in file:
            model_name = file.replace('_light_model_metadata.pickle', '')
            model_names.append(model_name)

    for model_name in model_names:
        _import_model_by_name(model_name)

    print('Model files loaded')


def _adapt_column(col_stats, col):
    icm = {
        'column_name': col,
        'data_type': col_stats['typing']['data_type'],
        'data_subtype': col_stats['typing']['data_subtype'],
    }

    icm['data_type_distribution'] = {
        'type': "categorical"
        ,'x': []
        ,'y': []
    }
    for k in col_stats['typing']['data_type_dist']:
        icm['data_type_distribution']['x'].append(k)
        icm['data_type_distribution']['y'].append(col_stats['typing']['data_type_dist'][k])

    icm['data_subtype_distribution'] = {
        'type': "categorical"
        ,'x': []
        ,'y': []
    }
    for k in col_stats['typing']['data_subtype_dist']:
        icm['data_subtype_distribution']['x'].append(k)
        icm['data_subtype_distribution']['y'].append(col_stats['typing']['data_subtype_dist'][k])

    icm['data_distribution'] = {}
    icm['data_distribution']['data_histogram'] = {
        "type": "categorical",
        'x': [],
        'y': []
    }
    icm['data_distribution']['clusters'] = [
         {
             "group": [],
             "members": []
         }
     ]

    for i in range(len(col_stats['histogram']['x'])):
        icm['data_distribution']['data_histogram']['x'].append(col_stats['histogram']['x'][i])
        icm['data_distribution']['data_histogram']['y'].append(col_stats['histogram']['y'][i])
    return icm


def get_model_data(model_name=None, lmd=None):
    if model_name is None and lmd is None:
        raise ValueError('provide either model name or lmd')

    if lmd is not None:
        pass
    elif model_name is not None:
        with MDBLock('shared', 'get_data_' + model_name):
            with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, f'{model_name}_light_model_metadata.pickle'), 'rb') as fp:
                lmd = pickle.load(fp)

    # ADAPTOR CODE
    amd = {}

    if 'stats_v2' in lmd:
        amd['data_analysis_v2'] = lmd['stats_v2']

    if lmd['current_phase'] == MODEL_STATUS_TRAINED:
        amd['status'] = 'complete'
    elif lmd['current_phase'] == MODEL_STATUS_ERROR:
        amd['status'] = 'error'
    else:
        amd['status'] = 'training'

    # Shared keys
    for k in ['name', 'version', 'is_active', 'data_source', 'predict', 'current_phase',
    'train_end_at', 'updated_at', 'created_at','data_preparation', 'validation_set_accuracy']:
        if k == 'predict':
            amd[k] = lmd['predict_columns']
        elif k in lmd:
            amd[k] = lmd[k]
            if k == 'validation_set_accuracy':
                if lmd['validation_set_accuracy'] is not None:
                    amd['accuracy'] = round(lmd['validation_set_accuracy'],3)
                else:
                    amd['accuracy'] = None
        else:
            amd[k] = None

    amd['data_analysis'] = {
        'target_columns_metadata': []
        ,'input_columns_metadata': []
    }

    amd['model_analysis'] = []

    for col in lmd['model_columns_map'].keys():
        if col in lmd['columns_to_ignore']:
            continue

        try:
            icm = _adapt_column(lmd['stats_v2'][col],col)
        except Exception as e:
            print(e)
            icm = {'column_name': col}
            #continue

        amd['force_vectors'] = {}
        if col in lmd['predict_columns']:
            # Histograms for plotting the force vectors
            if 'all_columns_prediction_distribution' in lmd and lmd['all_columns_prediction_distribution'] is not None:
                amd['force_vectors'][col] = {}
                amd['force_vectors'][col]['normal_data_distribution'] = lmd['all_columns_prediction_distribution'][col]
                amd['force_vectors'][col]['normal_data_distribution']['type'] = 'categorical'

                amd['force_vectors'][col]['missing_data_distribution'] = {}
                for missing_column in lmd['columnless_prediction_distribution'][col]:
                    amd['force_vectors'][col]['missing_data_distribution'][missing_column] = lmd['columnless_prediction_distribution'][col][missing_column]
                    amd['force_vectors'][col]['missing_data_distribution'][missing_column]['type'] = 'categorical'

                icm['importance_score'] = None
            amd['data_analysis']['target_columns_metadata'].append(icm)

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

                if lmd['columns_buckets_importances'] is not None and col in lmd['columns_buckets_importances']:
                    for output_col_bucket in lmd['columns_buckets_importances'][col]:
                        x_explained_member = []
                        for input_col in lmd['columns_buckets_importances'][col][output_col_bucket]:
                            stats = lmd['columns_buckets_importances'][col][output_col_bucket][input_col]
                            adapted_sub_incol = _adapt_column(stats, input_col)
                            x_explained_member.append(adapted_sub_incol)
                        mao['accuracy_histogram']['x_explained'].append(x_explained_member)

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
        else:
            if 'column_importances' in lmd and lmd['column_importances'] is not None:
                icm['importance_score'] = lmd['column_importances'][col]
            amd['data_analysis']['input_columns_metadata'].append(icm)

    return amd


def get_models():
    models = []
    for fn in os.listdir(CONFIG.MINDSDB_STORAGE_PATH):
        if '_light_model_metadata.pickle' in fn:
            model_name = fn.replace('_light_model_metadata.pickle','')
            try:
                amd = get_model_data(model_name)
                model = {}
                for k in ['name', 'version', 'is_active', 'data_source', 'predict',
                'status', 'train_end_at', 'updated_at', 'created_at','current_phase', 'accuracy']:
                    if k in amd:
                        model[k] = amd[k]
                    else:
                        model[k] = None

                models.append(model)
            except Exception as e:
                print(f"Can't adapt metadata for model: '{model_name}' when calling `get_models()`")
                raise

    return models
