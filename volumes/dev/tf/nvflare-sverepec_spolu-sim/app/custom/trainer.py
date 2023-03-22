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

# use GPU if previously disabled (-1)
import os
del os.environ['CUDA_VISIBLE_DEVICES']

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
import myModel3

from keras.utils.np_utils import to_categorical
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from wandb.keras import WandbMetricsLogger, WandbEvalCallback

PROJECT_NAME = 'FL-HYSPED-sverepec_spolu'

TF_FORCE_GPU_ALLOW_GROWTH = 'TF_FORCE_GPU_ALLOW_GROWTH'
TF_GPU_MEMORY_LIMIT ='TF_GPU_MEMORY_LIMIT'

module_logger = logging.getLogger(__name__)
module_logger.info('loading %s' % __name__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # configure utilization of GPUs
    try:
        for gpu in gpus:
            # memory growth setting
            if TF_FORCE_GPU_ALLOW_GROWTH in os.environ.keys():
                force_growth = os.environ[TF_FORCE_GPU_ALLOW_GROWTH].lower() == 'true'
                tf.config.experimental.set_memory_growth(gpu, force_growth)
                module_logger.info('%s %s: %s' % (gpu.name, TF_FORCE_GPU_ALLOW_GROWTH, force_growth))
            if TF_GPU_MEMORY_LIMIT in os.environ.keys():
                memory_limit = int(os.environ[TF_GPU_MEMORY_LIMIT])
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                module_logger.info('%s %s: %d' % (gpu.name, TF_GPU_MEMORY_LIMIT, memory_limit))
        logical_gpus = tf.config.list_logical_devices('GPU')
        module_logger.info('%d Physical GPUs, %d Logical GPUs' % (len(gpus), len(logical_gpus)))
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        module_logger.error(e)

class SimpleTrainer(Executor):
    
    WANDB_SITE_RUNS = 'WANDB_SITE_RUNS'
    
    def __init__(self, epochs_per_round):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs_per_round = epochs_per_round
        self.train_data, self.train_labels = None, None
        self.test_data, self.test_labels = None, None
        self.model = None
        self.var_list = None
        self.logger.info('__init__(epochs_per_round=%d)' % epochs_per_round)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        self.logger.info('%s\t%s\tevent_type: %s' % (fl_ctx.get_identity_name(), fl_ctx.get_job_id(), str(event_type)))
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)
        elif event_type == EventType.END_RUN:
            if self.wandb:
                self.wandb.finish()
    
    def setupWandB(self, fl_ctx: FLContext):
        if self.wandb is None:
            peer_context = fl_ctx.get_peer_context()
            self.log_info(fl_ctx, 'PEER_CONTEXT: %s' % str(peer_context))
            timestamp = peer_context.get_prop('JOB_START_TIMESTAMP')
            job_name = '%s-%s' % (fl_ctx.get_job_id(), timestamp)
            wandb_id = '%s-%s' % (job_name, fl_ctx.get_identity_name())
            self.log_info(fl_ctx, 'wandb_id: %s' % wandb_id)
            self.wandb = wandb.init(
                project=PROJECT_NAME,
                id=wandb_id,
                config={
                    'job_id': fl_ctx.get_job_id(),
                    'job_name': job_name,
                    'job_start_timestamp': timestamp,
                    'task_id': fl_ctx.get_prop(FLContextKey.TASK_ID, None),
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

    def setup(self, fl_ctx: FLContext):

        client_name = fl_ctx.get_identity_name()
        
        self.logger.info('setting up %s' % client_name)
        
        self.log_info(fl_ctx, 'cwd: %s' % os.getcwd())

        # load site's train data
        data_dir = os.path.join('..', '..', '..', 'data')
        filename_train = os.path.join(data_dir, client_name, 'train.csv')
        self.logger.info('loading %s train data: %s' % (client_name, filename_train))
        df_train = pd.read_csv(filename_train)
        
        # selection of attributes
        label_col = df_train.DRUH_DR.name
        self.train_data = df_train[df_train.columns.drop(label_col)]
        self.train_labels = df_train[label_col]
        # 1-hot encoding of labels
        self.train_labels = to_categorical(self.train_labels, num_classes=None)
        self.log_info(fl_ctx, 'train_data.shape: %s' % str(self.train_data.shape))
        self.log_info(fl_ctx, 'train_labels.shape: %s' % str(self.train_labels.shape))
                
        # load sites' test data
        filename_test = os.path.join(data_dir, client_name, 'test.csv')
        self.logger.info('loading %s test data: %s' % (client_name, filename_test))
        df_test = pd.read_csv(filename_test)
        self.test_data = df_test[df_test.columns.drop(label_col)]
        self.test_labels = df_test[label_col]
        self.test_labels = to_categorical(self.test_labels, num_classes=None)
        self.log_info(fl_ctx, 'test_data.shape: %s' % str(self.test_data.shape))
        self.log_info(fl_ctx, 'test_labels.shape: %s' % str(self.test_labels.shape))

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

        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != 'train':
            return make_reply(ReturnCode.TASK_UNKNOWN)
        
        
        # wandb
        self.setupWandB(fl_ctx)
        
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
