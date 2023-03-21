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

import logging
import numpy as np
import os
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

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

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
        
        # wand
        site_run_name = '%s-%s' % (fl_ctx.get_job_id(), fl_ctx.get_identity_name())
        self.site_run_name = site_run_name
        
        wandb_site_runs = fl_ctx.get_prop(SimpleTrainer.WANDB_SITE_RUNS, None)
        
        if wandb_site_runs is None:
            wandb_site_runs = {}
            fl_ctx.set_prop(SimpleTrainer.WANDB_SITE_RUNS, wandb_site_runs, private=True, sticky=True)
        else:
            self.log_info(fl_ctx, 'wandb_site_runs.keys(): %s' % ','.join([k for k in wandb_site_runs.keys()]))
            
        if site_run_name not in wandb_site_runs.keys():
            self.log_info(fl_ctx, 'creating run: %s' % site_run_name)
            wandb_site_runs[site_run_name] = {
                'run': wandb.init(
                    project=PROJECT_NAME,
                    config={
                        'job_id': fl_ctx.get_job_id(),
                        'identity_name': fl_ctx.get_identity_name(),
                        'runtime_args': fl_ctx.get_prop(FLContextKey.ARGS)
                    },
                    resume=True
                ),
                'callbacks': [
                    WandbMetricsLogger(),
                ]
            }
            fl_ctx.set_prop(SimpleTrainer.WANDB_SITE_RUNS, wandb_site_runs, private=True, sticky=True)
        
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
            callbacks=fl_ctx.get_prop(SimpleTrainer.WANDB_SITE_RUNS)[self.site_run_name]['callbacks']
        )

        # report updated weights in shareable
        weights = {k: v.numpy() for k,v in self.model.get_weight_paths().items()}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, 'Local epochs finished. Returning shareable')
        new_shareable = dxo.to_shareable()
        return new_shareable
