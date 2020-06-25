import os
import pickle
import shutil
import zipfile

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
