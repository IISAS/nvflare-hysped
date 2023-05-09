# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging

module_logger = logging.getLogger(__name__)
module_logger.info('loading %s' % __name__)
module_logger.setLevel(logging.DEBUG)

import myModel3
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb

from keras.utils.np_utils import to_categorical
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorClientRunManager
from tf2_common.tf2_constants import Constants as TF2Constants
from tf2_common.tf2_utils import Utils as TF2Utils
from wandb.keras import WandbMetricsLogger, WandbEvalCallback

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # configure utilization of GPUs
    for gpu in gpus:
        
        module_logger.info('physical GPU: %s' % str(gpu))
        
#        try:
#            # memory growth setting
#            if TF2Constants.TF_FORCE_GPU_ALLOW_GROWTH in os.environ.keys():
#                force_growth = os.environ[TF2Constants.TF_FORCE_GPU_ALLOW_GROWTH].lower() == 'true'
#                module_logger.info('setting %s=%s for %s ...' % (TF2Constants.TF_FORCE_GPU_ALLOW_GROWTH, force_growth, gpu.name))
#                tf.config.experimental.set_memory_growth(gpu, force_growth)
#                module_logger.info('%s %s: %s' % (gpu.name, TF2Constants.TF_FORCE_GPU_ALLOW_GROWTH, force_growth))
#        except Exception as e:
#            # Virtual devices must be set before GPUs have been initialized
#            module_logger.error(e)
            
        try:
            if TF2Constants.TF_GPU_MEMORY_LIMIT in os.environ.keys():
                memory_limit = int(os.environ[TF2Constants.TF_GPU_MEMORY_LIMIT])
                module_logger.info('setting %s=%s for %s ...' % (TF2Constants.TF_GPU_MEMORY_LIMIT, memory_limit, gpu.name))
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                module_logger.info('%s %s: %d' % (gpu.name, TF2Constants.TF_GPU_MEMORY_LIMIT, memory_limit))
        except Exception as e:
            # Virtual devices must be set before GPUs have been initialized
            module_logger.error(e)
            
    module_logger.info('%d Physical GPUs' % (len(gpus)))


try:
    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        # list logical GPUs
        for gpu in gpus:
            module_logger.info('logical GPU: %s' % str(gpu))
        module_logger.info('%d Logical GPUs' % (len(gpus)))
except Exception as e:
    module_logger.error(e)

class SimpleTrainer(Executor):
    
    WANDB_SITE_RUNS = 'WANDB_SITE_RUNS'
    
    def __init__(
        self,
        epochs_per_round,
        num_classes,
        label_col,
        wandb_key,
        project_name
    ):
        # Init functions of components should be very minimal. Init
        # is called when json is read. A big init will cause json loading to halt
        # for long time.
        super().__init__()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        self.epochs_per_round = epochs_per_round
        self.num_classes = num_classes
        self.label_col = label_col
        self.wandb_key = wandb_key
        self.project_name = project_name
        
        self.in_simulation = False
        self.wandb = None
        self.train_data, self.train_labels = None, None
        self.test_data, self.test_labels = None, None
        self.model = None
        self.var_list = None
        
        self.logger.info(
            '__init__(' \
            'epochs_per_round=%d, ' \
            'num_classes=%d, ' \
            'label_col=%s,' \
            'wandb_key=%s' \
            ')' % (
                epochs_per_round,
                num_classes,
                label_col,
                wandb_key != None
            )
        )
        
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        self.log_info(fl_ctx, 'event_type: %s' % str(event_type))
        if event_type == EventType.START_RUN:
            # Create all major components here.
            self._start_run_event(fl_ctx)
        elif event_type == EventType.END_RUN:
            # Clean up resources (closing files, joining threads, removing dirs etc.)
            self._end_run_event(fl_ctx)

    def _is_in_simulation(self):
        return self.in_simulation

    def _initWandB(self, fl_ctx: FLContext):

        if self.wandb is not None:
            # already initialized
            return

        self.log_info(fl_ctx, 'initializing WandB...')

        if not wandb.login(key=self.wandb_key):
            err_msg = 'could not login to WandB'
            self.log_error(fl_ctx, err_msg)
            raise Exception(err_msg)

        # resolve job name
        job_name = TF2Utils.get_peer_job_name(fl_ctx, default=fl_ctx.get_job_id())

        # resolve site's wandb_id
        wandb_id = fl_ctx.get_prop(TF2Constants.WANDB_ID, None)
        if wandb_id is None:
            self.log_info(fl_ctx, 'generating wandb_id...')
            wandb_id = TF2Utils.generate_wandb_id(
                job_name,
                fl_ctx.get_identity_name()
            )
            fl_ctx.set_prop(TF2Constants.WANDB_ID, wandb_id, private=True, sticky=True)
        self.log_info(fl_ctx, 'wandb_id: %s' % wandb_id)

        self.wandb = wandb.init(
            project=self.project_name,
            id=wandb_id,
            config={
                'job_id': fl_ctx.get_job_id(),
                'job_name': job_name,
                'job_start_timestamp': TF2Utils.get_peer_job_start_timestamp(fl_ctx, default='?'),
                'identity_name': fl_ctx.get_identity_name(),
                'runtime_args': fl_ctx.get_prop(FLContextKey.ARGS)
            },
            resume=True,
            reinit=True
        )

        # callbacks
        self.callbacks = [
            WandbMetricsLogger(),
        ]
    
    def _deinitWandB(self, fl_ctx: FLContext):
        if self.wandb is not None:
            self.log_info(fl_ctx, 'de-initializing WandB...')
            self.wandb.finish()
            self.log_info(fl_ctx, 'WandB finished')
    
    def _load_dataset(self, fl_ctx: FLContext, filename):
        self.log_info(fl_ctx, 'loading dataset: %s' % (filename))
        df = pd.read_csv(filename)
        # selection of attributes
        df_data = df[df.columns.drop(self.label_col)]
        df_labels = df[self.label_col]
        # 1-hot encoding of labels
        df_labels = to_categorical(df_labels, num_classes=self.num_classes)
        return df_data, df_labels
    
    def _load_datasets(self, fl_ctx: FLContext):
        
        engine = fl_ctx.get_engine()
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
        run_dir = engine.get_workspace().get_run_dir(job_id)
        self.log_info(fl_ctx, 'run_dir: %s' % os.path.abspath(run_dir))
        
        # resolve data dir
        data_dir = os.path.join(run_dir, '..', 'data')
        if self.in_simulation:
            client_name = fl_ctx.get_identity_name()
            data_dir = os.path.join(run_dir, '..', '..', 'data', fl_ctx.get_identity_name())
        self.log_info(fl_ctx, 'data_dir: %s' % os.path.abspath(data_dir))
        
        # train dataset
        self.train_data, self.train_labels = self._load_dataset(
            fl_ctx,
            os.path.join(data_dir, 'train.csv')
        )
        self.log_info(fl_ctx, 'train_data.shape: %s' % str(self.train_data.shape))
        self.log_info(fl_ctx, 'train_labels.shape: %s' % str(self.train_labels.shape))
        
        # test dataset
        self.test_data, self.test_labels = self._load_dataset(
            fl_ctx,
            os.path.join(data_dir, 'test.csv')
        )
        self.log_info(fl_ctx, 'test_data.shape: %s' % str(self.test_data.shape))
        self.log_info(fl_ctx, 'test_labels.shape: %s' % str(self.test_labels.shape))        
            
    def _start_run_event(self, fl_ctx: FLContext):

        self.log_info(fl_ctx, '_start_run_event')
        
        engine = fl_ctx.get_engine()
        if isinstance(engine, SimulatorClientRunManager):
            self.in_simulation = True
        
        self._load_datasets(fl_ctx)
            
        # infer the input shape from the train data
        input_shape = (self.train_data.shape[1],)
        self.log_info(fl_ctx, 'input_shape: %s' % str(input_shape))
        
        # infer the number of classes from the train data
        num_classes = self.train_labels.shape[1]
        self.log_info(fl_ctx, 'num_classes: %s' % str(num_classes))
        
        # compile a model
        model = myModel3.load_model(load_weights=False)

        self.var_list = [k for k,v in model.get_weight_paths().items()]
        self.log_info(fl_ctx, 'var_list: %s' % str(self.var_list))
        
        self.wandb = None       
        self.model = model
    
    def _end_run_event(self, fl_ctx):
        self._deinitWandB(fl_ctx)
        
    def _validate_task(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ):
        try:
            # First we extract DXO from the shareable.
            try:
                model_dxo = from_shareable(shareable)
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to extract model dxo from shareable. Exception: {e.__str__()}")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Get model from shareable. data_kind must be WEIGHTS.
            if model_dxo.data and model_dxo.data_kind == DataKind.WEIGHTS:
                model_weights = model_dxo.data
            else:
                self.log_error(
                    fl_ctx, "Model DXO doesn't have data or is not of type DataKind.WEIGHTS. Unable to validate."
                )
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # compile a model
            model = myModel3.load_model(load_weights=False)
            # TODO: layer filtering
            model.set_weights(list(model_weights.values()))

            # The workflow provides MODEL_OWNER information in the shareable header.
            model_name = shareable.get_header(AppConstants.MODEL_OWNER, "?")

            # Print properties.
            self.log_info(fl_ctx, f"Model: \n{model}")
            self.log_info(fl_ctx, f"Task name: {AppConstants.TASK_VALIDATION}")
            self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
            self.log_info(fl_ctx, f"Validating model from {model_name}.")

            # Check abort signal regularly.
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # Do validation.
            X = self.test_data
            Y_true = self.test_labels
            val_results = model.evaluate(X, Y_true)
            val_results = dict(zip(model.metrics_names, val_results))

            # Check abort signal regularly.
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.log_info(fl_ctx, f"Validation result: {val_results}")

            # Create DXO for metrics and return shareable.
            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in _validate_task: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
    def _submit_model_task(
        self,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ):
        try:
            weights = {k: v.numpy() for k,v in self.model.get_weight_paths().items()}
            # TODO: layer filterings
            model_dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
            return model_dxo.to_shareable()
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in _validate_task: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object received from server.
            fl_ctx: the `FLContext` object received from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """
        
        self.log_info(fl_ctx, 'execute: %s' % task_name)

        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        if task_name == AppConstants.TASK_VALIDATION:
            return self._validate_task(shareable, fl_ctx, abort_signal)
        
        if task_name == AppConstants.TASK_SUBMIT_MODEL:
            return self._submit_model_task(shareable, fl_ctx, abort_signal)

        if task_name != AppConstants.TASK_TRAIN:
            return make_reply(ReturnCode.TASK_UNKNOWN)

        # wandb
        self._initWandB(fl_ctx)
        
        dxo = from_shareable(shareable)
        model_weights = dxo.data
        # self.log_info(fl_ctx, 'nmodel_weights: %s' % str(model_weights))

        # use previous round's client weights to replace excluded layers from server
        prev_weights = {k: v.numpy() for k,v in self.model.get_weight_paths().items()}
        # self.log_info(fl_ctx, 'prev_weights: %s' % str(prev_weights))

        ordered_model_weights = {k: model_weights.get(k) for k in prev_weights}
        # self.log_info(fl_ctx, '\n\n\nordered_model_weights: %s' % str(ordered_model_weights))
        for key in self.var_list:
            value = ordered_model_weights.get(key)
            if np.all(value == 0):
                ordered_model_weights[key] = prev_weights[key]

        # update local model weights with received weights
        tmp = self.model.get_weights()
        self.model.set_weights(list(ordered_model_weights.values()))
        weights_diff = []
        for w in zip(tmp, self.model.get_weights()):
            weights_diff.append(w[0] - w[1])
        self.log_debug(fl_ctx, 'weights_diff:\n%s' % str(weights_diff))
        
        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(
            self.train_data,
            self.train_labels,
            epochs=self.epochs_per_round,
            validation_data=(self.test_data, self.test_labels),
            callbacks=self.callbacks
        )

        # report updated weights in shareable
        weights = {k: v.numpy() for k,v in self.model.get_weight_paths().items()}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, 'Local epochs finished. Returning shareable')
        new_shareable = dxo.to_shareable()
        return new_shareable
