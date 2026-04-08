## start vllm serving
### fix linux file issue
  ulimit -n 1048576

 export VLLM_USE_V1=0

 export VLLM_USE_FLASH_ATTN=0
 
 pip install flash-attn --no-build-isolation

vllm serve /workspace/vla/vlm_model_file/Qwen25-VL-7B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests

nohup vllm serve /workspace/vla/vlm_model_file/Qwen25-VL-7B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests \
    > vllm_judge.log 2>&1 &

[1] 104712

vllm serve /workspace/vla/vlm_model_file/Qwen3-VL-8B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests

## ray
ray stop

ray start --head \
  --num-gpus=4 \
  --dashboard-host=0.0.0.0 \
  --block

CUDA_VISIBLE_DEVICES=2,3 ray start --head \
  --num-gpus=2 \
  --dashboard-host=0.0.0.0 \
  --block

hostname -I

## training
export LLM_AS_A_JUDGE_BASE="http://10.1.5.26:18901/v1"

export LLM_AS_A_JUDGE_BASE="http://localhost:18901/v1"

export LLM_AS_A_JUDGE_BASE="http://127.0.0.1:18901/v1"

curl http://localhost:18901/v1/models

echo $LLM_AS_A_JUDGE_BASE

export WORLD_SIZE=8
export WORLD_SIZE=1

export SAVE_CHECKPOINT_DIR=/workspace/vla/vlm_model_file/verl_checkpoints

sh examples/agent/final_merged_v1v8_thinklite_h20.sh

agent:
  activate_agent: False
  single_response_max_tokens: 32768
  max_turns: 50
  concurrent_workers: 1
  tool_name_key: env_name
  tool_meta_key: null
  custom_stop: []
  show_tqdm: True
  vl_model_path: ${actor_rollout_ref.model.path}
  max_vllm_images: 32
  max_vllm_videos: 1

## 
ray start --head \
  --num-gpus=4 \
  --dashboard-host=0.0.0.0 \
  --block

sh /workspace/vla/DeepEyes-main/examples/agent/final_merged_v1v8_thinklite_h20_edx.sh

##

cd /workspace/vla/DeepEyes-main && ray start --head --num-gpus=8 --dashboard-host=0.0.0.0 > ray0218.log 2>&1 & sleep 5 && sh /workspace/vla/DeepEyes-main/examples/agent/final_merged_v1v8_thinklite_h20_edx.sh 2>&1 | tee /workspace/vla/DeepEyes-main/train0218.log

cd /workspace/vla/DeepEyes-main && ray start --head --num-gpus=2 > ray0221.log 2>&1 & sleep 5 && sh /workspace/vla/DeepEyes-main/examples/agent/final_merged_v1v8_thinklite_h20_2.sh 2>&1 | tee /workspace/vla/DeepEyes-main/train0221.log

cd /workspace/vla/DeepEyes-main && ray start --head --num-gpus=8 > ray0221.log 2>&1 & sleep 5 && sh /workspace/vla/DeepEyes-main/examples/agent/final_merged_v1v8_thinklite_h20_edx.sh 2>&1 | tee /workspace/vla/DeepEyes-main/train0221.log

## python env
   12  2026-02-18 14:31:31 pip install gymnasium
   21  2026-02-18 14:39:34 pip install duckduckgo_search
   24  2026-02-18 21:37:05 pip install playwright
   25  2026-02-18 21:37:46 pip install process_image
   95  2026-02-22 21:40:31 pip install flashinfer-python

## qwen25vl geo3k grpo training
cd /workspace/vla/DeepEyes-main && ray start --head --num-gpus=8 > ray0301.log 2>&1 & sleep 5 && sh /workspace/vla/DeepEyes-main/examples/grpo_trainer/run_qwen2_5_vl-7b-h20-edx.sh 2>&1 | tee /workspace/vla/DeepEyes-main/train0301_qwenvl25.log

cd /workspace/vla/DeepEyes-main && ray start --head --num-gpus=8 > ray0302.log 2>&1 & sleep 5 && sh /workspace/vla/DeepEyes-main/examples/grpo_trainer/run_qwen2_5_vl-7b-h20-edx2.sh 2>&1 | tee /workspace/vla/DeepEyes-main/train0302_qwenvl25.log

## merge hugging face weight

要自己新建权重路径 verl_checkpoints_0301_geo3k

python3 scripts/model_merger.py --local_dir /workspace/vla/DeepEyes-main/debug_for_single_node/global_step_64/actor --hf_model_path /workspace/vla/DeepEyes-main/debug_for_single_node/global_step_64/actor/huggingface

##
`step:1 / sample: 1033 / GT: Yes, the color of the leaves is red./ reward: 0.8`

`step:1 / sample: 1032 / GT: Yes, the color of the leaves is red./ reward: 0.8` 

1059	<think> The catcher is to the left of the batter in the image. The catcher's helmet and uniform can be seen, indicating their position to the left of the batter who is holding a baseball bat and wearing a red jersey and white pants. </think>  <tool_call> {""name"": ""image_zoom_in_tool"", ""arguments"": {""bbox_2d"": [0, 207, 140, 355], ""label"": ""the catcher""}} </tool_call> <answer> Yes, the catcher is to the left of the batter. </answer><|im_end|>"	0.800000011920929	Yes, the catcher is to the left of the batter.


<think>To determine if the leaves are red, I can zoom in on the area where the car appears to be located, as this will provide a closer view of the specific leaves in question.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [266, 101, 357, 219], "label": "leaves"}}
</tool_call>
<answer>The leaves appear to be red. The area is indeed zoomed in and shows red foliage.</answer><|im_end|>


<|im_start|>system
You are a helpful assistant.


# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call><|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>
Is the color of the leaves red?
Think first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> <|im_end|>
<|im_start|>assistant

<think>The image shows a person standing at the side of a road. There are trees with leaves visible in the background. To determine if the leaves are red, I would need to zoom in on the leaves of the trees to inspect their color. This can be done using the `image_zoom_in_tool` function.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [267, 133, 349, 212], "label": "the red leaves in the background"}}
</tool_call>
<answer>The leaves in the image are not red. They appear to be a darker shade, possibly brownish or reddish-brown, indicating they might be changing from green to winter coloration.</answer><|im_end|>




    