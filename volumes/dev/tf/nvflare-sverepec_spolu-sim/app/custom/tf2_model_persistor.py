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

import json
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import myModel3

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils import fobs

class TF2ModelPersistor(ModelPersistor):
    
    def __init__(self, save_name="tf2_model.fobs"):
        super().__init__()
        self.save_name = save_name
        self.data_dir = None
        self.input_shape = None
        self.num_classes = None

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        env = None
        run_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.log_info(fl_ctx, 'run_args: %s' % str(run_args))
        if run_args:
            env_config_file_name = os.path.join(app_root, run_args.env)
            if os.path.exists(env_config_file_name):
                try:
                    with open(env_config_file_name) as file:
                        env = json.load(file)
                except:
                    self.system_panic(
                        reason="error opening env config file {}".format(env_config_file_name), fl_ctx=fl_ctx
                    )
                    return

        if env is not None:
            if env.get("APP_CKPT_DIR", None):
                fl_ctx.set_prop(AppConstants.LOG_DIR, env["APP_CKPT_DIR"], private=True, sticky=True)
            if env.get("APP_CKPT") is not None:
                fl_ctx.set_prop(
                    AppConstants.CKPT_PRELOAD_PATH,
                    env["APP_CKPT"],
                    private=True,
                    sticky=True,
                )

        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            self.log_dir = app_root
        self._fobs_save_path = os.path.join(self.log_dir, self.save_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_info(fl_ctx, 'cwd: %s' % os.getcwd())
        self.data_dir = os.path.join('..', 'data')
        self.log_info(fl_ctx, 'data_dir: %s' % os.path.abspath(self.data_dir))

        # infer the input shape from the site-1 train data;
        # i.e., from the first row to skip loading whole data
        filename_train = os.path.join(self.data_dir, 'site-1', 'train.csv')
        df_train = pd.read_csv(filename_train, nrows=1)
        self.input_shape = (df_train.shape[1] - 1,)
        self.log_info(fl_ctx, 'infered input_shape: %s' % str(self.input_shape))

        # infer the number of classes from the label encoder
        label_encoder_filename = os.path.join(self.data_dir, 'encoder-DRUH_DR.npy')
        self.num_classes = np.load(label_encoder_filename, allow_pickle=True).shape[0]
        self.log_info(fl_ctx, 'infered num_classes: %s' % str(self.num_classes))

        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object
        """

        if os.path.exists(self._fobs_save_path):
            self.log_info(fl_ctx, 'Loading server weights ...')
            with open(self._fobs_save_path, "rb") as f:
                model_learnable = fobs.load(f)
        else:
            self.log_info(fl_ctx, 'Initializing server model ...')
                        
            # compile a model
            model = myModel3.load_model(load_weights=False)
            
            # layers <--> weights dictionary
            var_dict = {k: v.numpy() for k,v in model.get_weight_paths().items()}
            self.log_debug(fl_ctx, "var_dict: %s" % str(var_dict))
            
            model_learnable = make_model_learnable(var_dict, dict())
            
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        model_learnable_info = {k: str(type(v)) for k, v in model_learnable.items()}
        self.logger.info(f"Saving aggregated server weights: \n {model_learnable_info}")
        with open(self._fobs_save_path, "wb") as f:
            fobs.dump(model_learnable, f)
            
        # compile a model
        model = myModel3.load_model(load_weights=False)
        model.set_weights(list(model_learnable['weights'].values()))
        model.save(
            os.path.join(
                os.path.dirname(self._fobs_save_path),
                'model.h5'
            )
        )
