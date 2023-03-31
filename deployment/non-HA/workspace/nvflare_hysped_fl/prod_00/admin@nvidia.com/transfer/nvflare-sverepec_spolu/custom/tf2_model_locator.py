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

# do not use GPU at the FL server side (useful for running simulations)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model_locator import ModelLocator

from tensorflow import keras
from tf2_common.tf2_constants import Constants as TF2Constants
from typing import List

class TF2ModelLocator(ModelLocator):
    
    def __init__(
        self,
        model_dir = 'models',
        model_name = '%s.h5' % TF2Constants.GLOBAL_MODEL_NAME
    ):
        """The ModelLocator's job is to find the models to be included for cross site evaluation
        located on server. This TF2ModelLocator finds and extracts "server" model that is saved during training.

        Args:
            model_dir (str): Directory to look for models in. Defaults to "model"
            model_name (str). Name of the model. Defaults to "server.h5"
        """
        super().__init__()

        self.model_dir = model_dir
        self.model_file_name = model_name

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        """Returns the list of model names that should be included from server in cross site validation.add()

        Args:
            fl_ctx (FLContext): FL Context object.

        Returns:
            List[str]: List of model names.
        """
        return [TF2Constants.GLOBAL_MODEL_NAME]


    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        self.log_info(fl_ctx, 'locate_model(model_name=%s)' % model_name)
        dxo = None
        engine = fl_ctx.get_engine()

        if model_name == TF2Constants.GLOBAL_MODEL_NAME:
            try:
                job_id = fl_ctx.get_prop(FLContextKey.CURRENT_RUN)
                run_dir = engine.get_workspace().get_run_dir(job_id)
                model_path = os.path.join(run_dir, self.model_dir)

                model_load_path = os.path.join(model_path, self.model_file_name)
                model = None
                try:
                    model = keras.models.load_model(model_load_path)
                    self.log_info(fl_ctx, f"Loaded {model_name} model from {model_load_path}.")
                except Exception as e:
                    self.log_error(fl_ctx, f"Unable to load Keras Model: {e}.")

                if model is not None:
                    weights = {k: v.numpy() for k,v in model.get_weight_paths().items()}
                    dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={})
            except Exception as e:
                self.log_exception(fl_ctx, f"Exception in retrieving {TF2ModelLocator.SERVER_MODEL_NAME} model: {e}.")

        return dxo