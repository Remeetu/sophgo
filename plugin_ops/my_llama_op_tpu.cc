// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "func/llama_tpu.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> MyLlamaOp(const paddle::Tensor &x,
                                      const paddle::Tensor &y,
                                      const paddle::Tensor &z) {

  // std::string model_fn = "/workspace/PaddleCustomDevice/backends/sophgo/"
  //                        "plugin_ops/model/llama2-7b_int8_1dev.bmodel";
  std::vector<int> devices = {3};
  // std::string tokenizer_fn = "/workspace/PaddleCustomDevice/backends/sophgo/"
  //                            "plugin_ops/model/tokenizer.model";
  LLama2 llama;
  llama.init(devices);
  llama.chat();
  llama.deinit();

  return {paddle::add(x, paddle::add(y, z))};
}

std::vector<std::vector<int64_t>>
MyLamaOpInferShape(const std::vector<int64_t> &x_shape,
                   const std::vector<int64_t> &y_shape,
                   const std::vector<int64_t> &z_shape) {
  return {x_shape};
}

PD_BUILD_OP(my_llama)
    .Inputs({"X", "Y", "Z"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MyLlamaOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MyLamaOpInferShape)); // neccessary if the op has muti_inputs
