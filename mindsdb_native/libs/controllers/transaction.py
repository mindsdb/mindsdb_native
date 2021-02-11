from mindsdb_native.libs.constants.mindsdb import *
from mindsdb_native.libs.helpers.general_helpers import *
from mindsdb_native.libs.data_types.transaction_data import TransactionData
from mindsdb_native.libs.data_types.transaction_output_data import (
    PredictTransactionOutputData,
    TrainTransactionOutputData
)
from mindsdb_native.libs.data_types.mindsdb_logger import log
from mindsdb_native.config import CONFIG

from lightwood.api.predictor import Predictor

import _thread
import traceback
import importlib
import datetime
import pickle
import dill
import sys
from copy import deepcopy
import pandas as pd
import numpy as np


class BreakpointException(Exception):
    def __init__(self, ret):
        self.ret = ret


class Transaction:
    def __init__(self,
                 session,
                 light_transaction_metadata,
                 heavy_transaction_metadata,
                 logger=log):
        """
        A transaction is the interface to start some MindsDB operation within a session

        :param session:
        :param transaction_metadata: dict
        :param heavy_transaction_metadata: dict
        """

        self.session = session
        self.lmd = light_transaction_metadata
        self.lmd['created_at'] = str(datetime.datetime.now())
        self.hmd = heavy_transaction_metadata

        # variables to de defined by setup
        self.error = None
        self.errorMsg = None

        self.input_data = TransactionData()
        self.output_data = TrainTransactionOutputData()

        # variables that can be persisted

        self.log = logger

    def load_metadata(self):
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_STACK, [0x10000000, resource.RLIM_INFINITY])
            sys.setrecursionlimit(0x100000)
        except Exception:
            pass

        fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.lmd['name'], 'light_model_metadata.pickle')
        try:
            self.lmd = load_lmd(fn)
        except Exception as e:
            self.log.error(e)
            self.log.error(f'Could not load mindsdb light metadata from the file: {fn}')

        fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.hmd['name'], 'heavy_model_metadata.pickle')
        try:
            self.hmd = load_hmd(fn)
        except Exception as e:
            self.log.error(e)
            self.log.error(f'Could not load mindsdb heavy metadata in the file: {fn}')

        icp_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.hmd['name'], 'icp.pickle')
        try:
            with open(icp_fn, 'rb') as fp:
                self.hmd['icp'] = dill.load(fp)
                # restore MDB predictors in ICP objects
                for col in self.lmd['predict_columns']:
                    try:
                        self.hmd['icp'][col].nc_function.model.model = self.session.transaction.model_backend.predictor
                    except AttributeError:
                        model_path = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.hmd['name'], 'lightwood_data')
                        self.hmd['icp'][col].nc_function.model.model = Predictor(load_from_path=model_path)

                    # restore model in normalizer
                    if self.hmd['icp'][col].nc_function.normalizer is not None:
                        self.hmd['icp'][col].nc_function.normalizer.model = self.hmd['icp'][col].nc_function.model.model

        except FileNotFoundError as e:
            self.hmd['icp'] = {'active': False}
            self.log.warning(f'Could not find mindsdb conformal predictor.')

        except Exception as e:
            self.log.error(e)
            self.log.error(f'Could not load mindsdb conformal predictor in the file: {icp_fn}')

    def save_metadata(self):
        Path(CONFIG.MINDSDB_STORAGE_PATH).joinpath(self.lmd['name']).mkdir(mode=0o777, exist_ok=True, parents=True)
        fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.lmd['name'], 'light_model_metadata.pickle')
        self.lmd['updated_at'] = str(datetime.datetime.now())
        try:
            with open(fn, 'wb') as fp:
                pickle.dump(self.lmd, fp,protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.log.error(traceback.format_exc())
            self.log.error(e)
            self.log.error(f'Could not save mindsdb light metadata in the file: {fn}')

        fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.hmd['name'], 'heavy_model_metadata.pickle')
        save_hmd = {}
        null_out_fields = ['from_data', 'icp', 'breakpoint','sample_function']
        for k in null_out_fields:
            save_hmd[k] = None

        for k in self.hmd:
            if k not in null_out_fields:
                save_hmd[k] = self.hmd[k]
            if k == 'model_backend' and not isinstance(self.hmd['model_backend'], str):
                save_hmd[k] = None

        try:
            with open(fn, 'wb') as fp:
                # Don't save data for now
                pickle.dump(save_hmd, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            self.log.error(e)
            self.log.error(traceback.format_exc())
            self.log.error(f'Could not save mindsdb heavy metadata in the file: {fn}')

        if 'icp' in self.hmd.keys() and self.hmd['icp']['active']:
            icp_fn = os.path.join(CONFIG.MINDSDB_STORAGE_PATH, self.hmd['name'], 'icp.pickle')
            try:
                mdb_predictors = {}
                with open(icp_fn, 'wb') as fp:
                    # clear data cache
                    for key in self.hmd['icp'].keys():
                        if key != 'active':
                            mdb_predictors[key] = self.hmd['icp'][key].nc_function.model.model
                            self.hmd['icp'][key].nc_function.model.model = None
                            self.hmd['icp'][key].nc_function.model.last_x = None
                            self.hmd['icp'][key].nc_function.model.last_y = None
                            if self.hmd['icp'][key].nc_function.normalizer is not None:
                                self.hmd['icp'][key].nc_function.normalizer.model = None

                    dill.dump(self.hmd['icp'], fp, protocol=dill.HIGHEST_PROTOCOL)

                    # restore predictor in ICP
                    for key in self.hmd['icp'].keys():
                        if key != 'active':
                            self.hmd['icp'][key].nc_function.model.model = mdb_predictors[key]

            except Exception as e:
                self.log.error(e)
                self.log.error(traceback.format_exc())
                self.log.error(f'Could not save mindsdb conformal predictor in the file: {icp_fn}')

    def _call_phase_module(self, module_name, **kwargs):
        """
        Loads the module and runs it
        """

        self.lmd['is_active'] = True
        self.lmd['phase'] = module_name
        module_path = convert_cammelcase_to_snake_string(module_name)
        module_full_path = f'mindsdb_native.libs.phases.{module_path}.{module_path}'
        try:
            main_module = importlib.import_module(module_full_path)
            module = getattr(main_module, module_name)
            ret = module(self.session, self)(**kwargs)
        except Exception:
            error = f'Could not load module {module_name}'
            self.log.error(error)
            raise
        else:
            if isinstance(self.hmd['breakpoint'], str):
                if module_name == self.hmd['breakpoint']:
                    raise BreakpointException(ret=ret)
            elif isinstance(self.hmd['breakpoint'], dict):
                if module_name in self.hmd['breakpoint']:
                    if callable(self.hmd['breakpoint'][module_name]):
                        self.hmd['breakpoint'][module_name]()
                    else:
                        raise ValueError('breakpoint dict must have callable values')
            return ret
        finally:
            self.lmd['phase'] = module_name
            self.lmd['is_active'] = False

    def run(self):
        raise NotImplementedError


class MutatingTransaction(Transaction):
    def run(self, mutating_callback):
        self.load_metadata()
        mutating_callback(self.lmd, self.hmd)
        self.save_metadata()

class LearnTransaction(Transaction):
    def _run(self):
        try:
            self.lmd['current_phase'] = MODEL_STATUS_PREPARING
            self.save_metadata()

            self._call_phase_module(module_name='DataExtractor')
            self.save_metadata()

            self._call_phase_module(module_name='DataCleaner')
            self.save_metadata()

            self._call_phase_module(module_name='TypeDeductor',
                                    input_data=self.input_data)
            self.save_metadata()

            self.lmd['current_phase'] = MODEL_STATUS_DATA_ANALYSIS
            self._call_phase_module(module_name='DataAnalyzer',
                                    input_data=self.input_data)
            self.save_metadata()

            # quick_learn can still be set to False explicitly to disable this behavior
            if self.lmd['quick_learn'] is None:
                n_cols = len(self.lmd['columns'])
                n_cells = n_cols * self.lmd['data_preparation']['used_row_count']
                if n_cols >= 80 and n_cells > int(1e5):
                    self.log.warning('Data has too many columns, disabling column importance feature')
                    self.lmd['disable_column_importance'] = True

            self._call_phase_module(module_name='DataCleaner')
            self.save_metadata()

            self._call_phase_module(module_name='DataSplitter')
            self.save_metadata()

            self._call_phase_module(module_name='DataTransformer', input_data=self.input_data)
            self.lmd['current_phase'] = MODEL_STATUS_TRAINING
            self.save_metadata()
            self._call_phase_module(module_name='ModelInterface', mode='train')

            if self.lmd['quick_learn']:
                predict_method = self.session.predict
                def predict_method_wrapper(*args, **kwargs):
                    if 'advanced_args' not in kwargs:
                        kwargs['advanced_args'] = {}
                    kwargs['advanced_args']['quick_predict'] = True
                    return predict_method(*args, **kwargs)
                self.session.predict = predict_method_wrapper
            else:
                self.lmd['current_phase'] = MODEL_STATUS_ANALYZING
                self.save_metadata()
                self._call_phase_module(module_name='ModelAnalyzer')

            self.lmd['current_phase'] = MODEL_STATUS_TRAINED
            self.save_metadata()
            return

        except Exception as e:
            self.lmd['is_active'] = False
            self.lmd['current_phase'] = MODEL_STATUS_ERROR
            self.lmd['stack_trace_on_error'] = traceback.format_exc()
            # Come  up with a function that tries to explain the error in a more human readable~ish way
            self.lmd['error_explanation'] = str(e)
            self.log.error(str(e))
            self.save_metadata()
            raise e

    def run(self):
        if CONFIG.EXEC_LEARN_IN_THREAD == False:
            self._run()
        else:
            _thread.start_new_thread(self._run(), ())

class AnalyseTransaction(Transaction):
    def run(self):
        self._call_phase_module(module_name='DataExtractor')
        self._call_phase_module(module_name='DataCleaner')
        self._call_phase_module(module_name='TypeDeductor', input_data=self.input_data)
        self._call_phase_module(module_name='DataAnalyzer', input_data=self.input_data)
        self.lmd['current_phase'] = MODEL_STATUS_DONE

class PredictTransaction(Transaction):
    def run(self):
        old_lmd = {}
        for k in self.lmd: old_lmd[k] = self.lmd[k]

        old_hmd = {}
        for k in self.hmd: old_hmd[k] = self.hmd[k]
        self.load_metadata()

        for k in old_lmd:
            if old_lmd[k] is not None:
                self.lmd[k] = old_lmd[k]
            else:
                if k not in self.lmd:
                    self.lmd[k] = None

        for k in old_hmd:
            if old_hmd[k] is not None:
                self.hmd[k] = old_hmd[k]
            else:
                if k not in self.hmd:
                    self.hmd[k] = None

        if self.lmd is None:
            self.log.error('No metadata found for this model')
            return

        self._call_phase_module(module_name='DataExtractor')

        if self.input_data.data_frame.shape[0] <= 0:
            self.log.error('No input data provided !')
            return

        if self.lmd['tss']['is_timeseries']:
            self._call_phase_module(module_name='DataSplitter')

        self._call_phase_module(module_name='DataTransformer', input_data=self.input_data)

        self._call_phase_module(module_name='ModelInterface', mode='predict')

        if self.lmd['return_raw_predictions']:
            self.output_data = PredictTransactionOutputData(
                transaction=self,
                data=self.hmd['predictions']
            )
            return self.output_data

        output_data = {col: [] for col in self.lmd['columns']}

        if 'make_predictions' in self.input_data.data_frame.columns:
            predictions_df = pd.DataFrame(
                self.input_data.data_frame[
                    self.input_data.data_frame['make_predictions'] == True
                ]
            )
            del predictions_df['make_predictions']
        else:
            predictions_df = self.input_data.data_frame

        for col in self.lmd['columns']:
            if col in self.lmd['predict_columns']:
                output_data[f'__observed_{col}'] = list(predictions_df[col])
                output_data[col] = self.hmd['predictions'][col]

                if f'{col}_class_distribution' in self.hmd['predictions']:
                    output_data[f'{col}_class_distribution'] = self.hmd['predictions'][f'{col}_class_distribution']
                    self.lmd['lightwood_data'][f'{col}_class_map'] = self.lmd['stats_v2'][col]['lightwood_class_map']
            else:
                output_data[col] = list(predictions_df[col])

        # confidence estimation
        if self.hmd['icp']['active'] and not self.lmd['quick_predict']:
            self.lmd['all_conformal_ranges'] = {}
            icp_X = deepcopy(predictions_df)

            if self.lmd['tss']['is_timeseries']:
                icp_X, _, _, _ = self.model_backend._ts_reshape(icp_X)  # TODO: avoid inefficient reshaping

            for col in self.lmd['columns_to_ignore'] + self.lmd['predict_columns']:
                if col in icp_X.columns:
                    icp_X.pop(col)

            for predicted_col in self.lmd['predict_columns']:
                output_data[predicted_col] = list(self.hmd['predictions'][predicted_col])
                output_data[f'{predicted_col}_confidence'] = [None] * len(output_data[predicted_col])
                output_data[f'{predicted_col}_confidence_range'] = [[None, None]] * len(output_data[predicted_col])

                if self.hmd['icp'].get(predicted_col, False):
                    typing_info = self.lmd['stats_v2'][predicted_col]['typing']
                    X = deepcopy(icp_X)

                    # preserve order that the ICP expects, else bounds are useless
                    X = X.reindex(columns=self.hmd['icp'][predicted_col].index.values)

                    normalizer = self.hmd['icp'][predicted_col].nc_function.normalizer
                    if normalizer:
                        normalizer.prediction_cache = self.hmd['predictions']

                    # numerical
                    if typing_info['data_type'] == DATA_TYPES.NUMERIC or \
                            (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                                DATA_TYPES.NUMERIC in typing_info['data_type_dist'].keys()):
                        std_tol = 1
                        tolerance = self.lmd['stats_v2'][predicted_col]['train_std_dev'] * std_tol
                        if self.lmd['tss']['is_timeseries'] and self.lmd['tss']['nr_predictions'] > 1:
                            # bounds in time series are only given for the first forecast
                            self.hmd['icp'][predicted_col].nc_function.model.prediction_cache = \
                                [p[0] for p in output_data[predicted_col]]
                        else:
                            self.hmd['icp'][predicted_col].nc_function.model.prediction_cache = output_data[predicted_col]
                        self.lmd['all_conformal_ranges'][predicted_col] = self.hmd['icp'][predicted_col].predict(X.values)

                        for sample_idx in range(self.lmd['all_conformal_ranges'][predicted_col].shape[0]):
                            sample = self.lmd['all_conformal_ranges'][predicted_col][sample_idx, :, :]
                            for idx in range(sample.shape[1]):
                                significance = (99 - idx) / 100
                                diff = sample[1, idx] - sample[0, idx]
                                if diff <= tolerance:
                                    output_data[f'{predicted_col}_confidence'][sample_idx] = significance
                                    conf_range = list(sample[:, idx])

                                    # for positive numerical domains
                                    if self.lmd['stats_v2'][predicted_col].get('positive_domain', False):
                                        conf_range[0] = max(0, conf_range[0])
                                    output_data[f'{predicted_col}_confidence_range'][sample_idx] = conf_range
                                    break
                            else:
                                output_data[f'{predicted_col}_confidence'][sample_idx] = 0.9901  # default
                                bounds = sample[:, 0]
                                sigma = (bounds[1] - bounds[0]) / 2
                                output_data[f'{predicted_col}_confidence_range'][sample_idx] = [bounds[0] - sigma, bounds[1] + sigma]
                    # categorical
                    elif typing_info['data_type'] == DATA_TYPES.CATEGORICAL or \
                            (typing_info['data_type'] == DATA_TYPES.SEQUENTIAL and
                                DATA_TYPES.CATEGORICAL in typing_info['data_type_dist'].keys()):
                        if self.lmd['stats_v2'][predicted_col]['typing']['data_subtype'] != DATA_SUBTYPES.TAGS:
                            significances = list(range(20)) + list(range(20, 100, 10))  # max permitted error rate
                            self.hmd['icp'][predicted_col].nc_function.model.prediction_cache = output_data[f'{predicted_col}_class_distribution']
                            all_ranges = np.array(
                                [self.hmd['icp'][predicted_col].predict(X.values, significance=s / 100)
                                    for s in significances])
                            self.lmd['all_conformal_ranges'][predicted_col] = np.swapaxes(np.swapaxes(all_ranges, 0, 2), 0, 1)

                            for sample_idx in range(self.lmd['all_conformal_ranges'][predicted_col].shape[0]):
                                sample = self.lmd['all_conformal_ranges'][predicted_col][sample_idx, :, :]
                                for idx in range(sample.shape[1]):
                                    conf = (99 - significances[idx]) / 100
                                    if np.sum(sample[:, idx]) == 1:
                                        output_data[f'{predicted_col}_confidence'][sample_idx] = conf
                                        break
                                else:
                                    output_data[f'{predicted_col}_confidence'][sample_idx] = 0.005
        else:
            for predicted_col in self.lmd['predict_columns']:
                output_data[f'{predicted_col}_confidence'] = [None] * len(output_data[predicted_col])
                output_data[f'{predicted_col}_confidence_range'] = [[None, None]] * len(output_data[predicted_col])

        self.output_data = PredictTransactionOutputData(
            transaction=self,
            data=output_data
        )


class BadTransaction(Transaction):
    def run(self):
        self.log.error(self.errorMsg)
        self.error = True
