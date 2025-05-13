from mmengine.config import read_base
from opencompass.models.openai_api import OpenAISDK

with read_base():
    from .lark import lark_bot_url
    # 直接从预设的数据集配置中读取所需的数据集配置
    # from .datasets.aime2025.aime2025_llmjudge_gen_5e9f4f import aime2025_datasets
    from .datasets.livecodebench.livecodebench_gen import LCB_datasets
    from .datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from .datasets.math.math_gen import math_datasets
    from .datasets.cmmlu.cmmlu_gen import cmmlu_datasets

datasets = (
        # aime2025_datasets +
        LCB_datasets +
        mmlu_pro_datasets +
        math_datasets +
        cmmlu_datasets
)

internlm_url = "http://v2.open.venus.oa.com/llmproxy"
internlm_api_key = "uiiMumeqx2RleTVgQwE3U6ZY@4133"

# done ✅
# gpt_4o_mini = [
#     dict(
#         abbr="gpt-4o-mini",
#         type=OpenAISDK,
#         path="gpt-4o-mini",               # 请求服务时的 model name
#         key=internlm_api_key,
#         openai_api_base=internlm_url,
#         rpm_verbose=True,                   # 是否打印请求速率
#         query_per_second=0.16,              # 服务请求速率
#         max_out_len=16384,                   # 最大输出长度
#         max_seq_len=16384,                   # 最大输入长度
#         temperature=0.7,                   # 生成温度
#         batch_size=1,                       # 批处理大小
#         retry=3,                            # 重试次数
#         run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
#     )
# ]
# models = gpt_4o_mini

# done ✅
# gemini_2_flash = [
#     dict(
#         abbr="gemini-2.0-flash",
#         type=OpenAISDK,
#         path="gemini-2.0-flash",               # 请求服务时的 model name
#         key=internlm_api_key,
#         openai_api_base=internlm_url,
#         rpm_verbose=True,                   # 是否打印请求速率
#         query_per_second=0.16,              # 服务请求速率
#         max_out_len=8192,                   # 最大输出长度
#         max_seq_len=16384,                   # 最大输入长度
#         temperature=0.7,                   # 生成温度
#         batch_size=1,                       # 批处理大小
#         retry=3,                            # 重试次数
#         run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
#     )
# ]
# models = gemini_2_flash

# done ✅
# gpt_4_1 = [
#     dict(
#         abbr="GPT-4.1",
#         type=OpenAISDK,
#         path="gpt-4.1",               # 请求服务时的 model name
#         key=internlm_api_key,
#         openai_api_base=internlm_url,
#         rpm_verbose=True,                   # 是否打印请求速率
#         query_per_second=0.16,              # 服务请求速率
#         max_out_len=16384,                   # 最大输出长度
#         max_seq_len=16384,                   # 最大输入长度
#         temperature=0.7,                   # 生成温度
#         batch_size=20,                       # 批处理大小
#         retry=3,                            # 重试次数
#         run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
#     )
# ]
# models = gpt_4_1

# done ✅
# qwen3_30b_fp8 = [
#     dict(
#         abbr="Qwen3-30B-A3B-FP8",
#         type=OpenAISDK,
#         path="Qwen/Qwen3-30B-A3B-FP8",               # 请求服务时的 model name
#         key="EMPTY",
#         openai_api_base="http://101.42.115.58:8000/v1/",
#         rpm_verbose=True,                   # 是否打印请求速率
#         query_per_second=0.16,              # 服务请求速率
#         max_out_len=40960,                   # 最大输出长度
#         max_seq_len=40960,                   # 最大输入长度
#         temperature=0.7,                   # 生成温度
#         batch_size=16,                       # 批处理大小
#         retry=3,                            # 重试次数
#         run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
#     )
# ]
# models = qwen3_30b_fp8

# done ✅
# openai_o4_mini = [
#     dict(
#         abbr="openai-o4-mini",
#         type=OpenAISDK,
#         path="o4-mini",               # 请求服务时的 model name
#         key=internlm_api_key,
#         openai_api_base=internlm_url,
#         rpm_verbose=True,                   # 是否打印请求速率
#         query_per_second=0.16,              # 服务请求速率
#         max_out_len=16384,                   # 最大输出长度
#         max_seq_len=16384,                   # 最大输入长度
#         temperature=1,                   # 生成温度
#         batch_size=20,                       # 批处理大小
#         retry=3,                            # 重试次数
#         run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
#     )
# ]
# models = openai_o4_mini

# doing 🚀
deepseek_r1 = [
    dict(
        abbr="deepseek-r1",
        type=OpenAISDK,
        path="deepseek-r1-local-II",               # 请求服务时的 model name
        key=internlm_api_key,
        openai_api_base=internlm_url,
        rpm_verbose=True,                   # 是否打印请求速率
        query_per_second=0.16,              # 服务请求速率
        max_out_len=40960,                   # 最大输出长度
        max_seq_len=40960,                   # 最大输入长度
        temperature=0.7,                   # 生成温度
        batch_size=1,                       # 批处理大小
        retry=3,                            # 重试次数
        run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
    )
]
models = deepseek_r1

# no start ❌
# qwen3_235b_a22b_fp8 = [
#     dict(
#         abbr="Qwen3-235B-A22B-FP8",
#         type=OpenAISDK,
#         path="qwen3-235b-a22b-fp8",               # 请求服务时的 model name
#         key=internlm_api_key,
#         openai_api_base=internlm_url,
#         rpm_verbose=True,                   # 是否打印请求速率
#         query_per_second=0.16,              # 服务请求速率
#         max_out_len=40960,                   # 最大输出长度
#         max_seq_len=40960,                   # 最大输入长度
#         temperature=0.7,                   # 生成温度
#         batch_size=1,                       # 批处理大小
#         retry=3,                            # 重试次数
#         run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
#     )
# ]
# models = qwen3_235b_a22b_fp8

work_dir = 'outputs/default/'
station_path = "/Users/zhangyuehua/Desktop/opencompass-result"

"""
python run.py opencompass/configs/llm_eval_standard_benchmark.py --debug -r 20250503_222944 -l --read-from-station
"""