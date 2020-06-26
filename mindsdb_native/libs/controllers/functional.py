import os
import pickle
import shutil
import zipfile

from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.config import CONFIG

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
        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name + '_light_model_metadata.pickle'), 'rb') as fp:
            lmd = pickle.load(fp)

        if 'ludwig_data' in lmd and 'ludwig_save_path' in lmd['ludwig_data']:
            lmd['ludwig_data']['ludwig_save_path'] = str(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,os.path.basename(lmd['ludwig_data']['ludwig_save_path'])))

        if 'lightwood_data' in lmd and 'save_path' in lmd['lightwood_data']:
            lmd['lightwood_data']['save_path'] = str(os.path.join(CONFIG.MINDSDB_STORAGE_PATH,os.path.basename(lmd['lightwood_data']['save_path'])))

        with open(os.path.join(CONFIG.MINDSDB_STORAGE_PATH, model_name + '_light_model_metadata.pickle'), 'wb') as fp:
            pickle.dump(lmd, fp,protocol=pickle.HIGHEST_PROTOCOL)
    print('Model files loaded')


def _adapt_column(col_stats, col):
    icm = {
        'column_name': col,
        'data_type': col_stats['data_type'],
        'data_subtype': col_stats['data_subtype'],
    }

    icm['data_type_distribution'] = {
        'type': "categorical"
        ,'x': []
        ,'y': []
    }
    for k in col_stats['data_type_dist']:
        icm['data_type_distribution']['x'].append(k)
        icm['data_type_distribution']['y'].append(col_stats['data_type_dist'][k])

    icm['data_subtype_distribution'] = {
        'type': "categorical"
        ,'x': []
        ,'y': []
    }
    for k in col_stats['data_subtype_dist']:
        icm['data_subtype_distribution']['x'].append(k)
        icm['data_subtype_distribution']['y'].append(col_stats['data_subtype_dist'][k])

    icm['data_distribution'] = {}
    icm['data_distribution']['data_histogram'] = {
        "type": "categorical",
        'x': [],
        'y': []
    }
    icm['data_distribution']['clusters'] =  [
         {
             "group": [],
             "members": []
         }
     ]


    for i in range(len(col_stats['histogram']['x'])):
        icm['data_distribution']['data_histogram']['x'].append(col_stats['histogram']['x'][i])
        icm['data_distribution']['data_histogram']['y'].append(col_stats['histogram']['y'][i])

    scores = ['consistency_score', 'redundancy_score', 'variability_score']
    for score in scores:
        metrics = []
        if score == 'consistency_score':
            simple_description = "A low value indicates the data is not very consistent, it's either missing a lot of valus or the type (e.g. number, text, category, date) of values varries quite a lot."
            metrics.append({
                  "type": "score",
                  "name": "Type Distribution",
                  "score": col_stats['data_type_distribution_score'],
                  "description": "A low value indicates that we can't consistently determine a single data type (e.g. number, text, category, date) for most values in this column",
                  "warning": col_stats['data_type_distribution_score_warning']
            })
            metrics.append({
                  "type": "score",
                  "score": col_stats['empty_cells_score'],
                  "name": "Empty Cells",
                  "description": "A low value indicates that a lot of the values in this column are empty or null. A value of 10 means no cell is missing data, a value of 0 means no cell has any data.",
                  "warning": col_stats['empty_cells_score_warning']
            })
            if 'duplicates_score' in col_stats:
                metrics.append({
                      "type": "score",
                      "name": "Value Duplication",
                      "score": col_stats['duplicates_score'],
                      "description": "A low value indicates that a lot of the values in this columns are duplicates, as in, the same value shows up more than once in the column. This is not necessarily bad and could be normal for certain data types.",
                      "warning": col_stats['duplicates_score_warning']
                })

        if score == 'variability_score':
            simple_description = "A low value indicates a high possibility of some noise affecting your data collection process. This could mean that the values for this column are not collected or processed correctly."
            if 'lof_based_outlier_score' in col_stats and 'z_test_based_outlier_score' in col_stats:
                metrics.append({
                      "type": "score",
                      "name": "Z Outlier Score",
                      "score": col_stats['lof_based_outlier_score'],
                      "description": "A low value indicates a large number of outliers in your dataset. This is based on distance from the center of 20 clusters as constructed via KNN.",
                      "warning": col_stats['lof_based_outlier_score_warning']
                })
                metrics.append({
                      "type": "score",
                      "name": "Z Outlier Score",
                      "score": col_stats['z_test_based_outlier_score'],
                      "description": "A low value indicates a large number of data points are more than 3 standard deviations away from the mean value of this column. This means that this column likely has a large amount of outliers",
                      "warning": col_stats['z_test_based_outlier_score_warning']
                })
            metrics.append({
                  "type": "score",
                  "name":"Value Distribution",
                  "score": col_stats['value_distribution_score'],
                  "description": "A low value indicates the possibility of a large number of outliers, the clusters in which your data is distributed aren't evenly sized.",
                  "warning": col_stats['value_distribution_score_warning']
            })

        if score == 'redundancy_score':
            # CLF based score to be included here once we find a faster way of computing it...
            similarity_score_based_most_correlated_column = col_stats['most_similar_column_name']

            simple_description = f"A low value indicates that the data in this column is highly redundant (useless) for making any sort of prediction. You should make sure that values heavily related to this column are not already expressed in the \"{similarity_score_based_most_correlated_column}\" column (e.g. if this column is a timestamp, make sure you don't have another column representing the exact same time in ISO datetime format)"

            metrics.append({
                  "type": "score",
                  "name": "Matthews Correlation Score",
                  "score": col_stats['similarity_score'],
                  "description": f"A low value indicates a large number of values in this column are similar to values in the \"{similarity_score_based_most_correlated_column}\" column",
                  "warning": col_stats['similarity_score_warning']
            })

        icm[score.replace('_score','')] = {
            "score": col_stats[score],
            "metrics": metrics,
            "description": simple_description,
            "warning": col_stats[f'{score}_warning']
        }

    return icm

def get_model_data(model_name, lmd=None):
    if lmd is None:
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
            icm = _adapt_column(lmd['column_stats'][col],col)
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
                print(e)
                print(traceback.format_exc())
                print(f"Can't adapt metadata for model: '{model_name}' when calling `get_models()`")

    return models
