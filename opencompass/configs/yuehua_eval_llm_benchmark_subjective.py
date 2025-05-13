from mmengine.config import read_base
from opencompass.models import OpenAISDK
from opencompass.models import TurboMindModelwithChatTemplate
from opencompass.datasets import CustomDataset
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import generic_llmjudge_postprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

# 导入评判模型配置
with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_14b_instruct import (
        models as judge_model,
    )
    from opencompass.configs.models.openai.gpt_4 import models as j
    # 主观
    from .datasets.subjective.arena_hard import arena_hard_compare

# 定义评判模板
JUDGE_TEMPLATE = """
请评估以下回答是否正确地回答了问题。
问题：{problem}
参考答案：{answer}
模型回答：{prediction}

模型回答是否正确？如果正确，请回答"A"；如果不正确，请回答"B"。
""".strip()

# 数据集读取配置
reader_cfg = dict(input_columns=['problem'], output_column='answer')

# 被评估模型的推理配置
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# 使用LLM评判器的评估配置
eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                begin=[
                    dict(
                        role='SYSTEM',
                        fallback_role='HUMAN',
                        prompt="你是一个负责评估模型输出正确性和质量的助手。",
                    )
                ],
                round=[
                    dict(role='HUMAN', prompt=JUDGE_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            type=CustomDataset,
            path='path/to/your/dataset',
            file_name='your_dataset.jsonl',
            reader_cfg=reader_cfg,
        ),
        judge_cfg=judge_model[0],
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)


datasets = (
    arena_hard_compare
)

internlm_url = "http://v2.open.venus.oa.com/llmproxy"
internlm_api_key = "uiiMumeqx2RleTVgQwE3U6ZY@4133"

gpt_4o_mini = [
    dict(
        type=OpenAISDK,
        path="gpt-4o-mini",               # 请求服务时的 model name
        key=internlm_api_key,
        openai_api_base=internlm_url,
        rpm_verbose=True,                   # 是否打印请求速率
        query_per_second=0.16,              # 服务请求速率
        max_out_len=16384,                   # 最大输出长度
        max_seq_len=16384,                   # 最大输入长度
        temperature=0.7,                   # 生成温度
        batch_size=1,                       # 批处理大小
        retry=3,                            # 重试次数
        run_cfg=dict(num_gpus=0),                # 资源需求（不需要 GPU）
        generation_kwargs=dict(
            do_sample=True,
        ),
    )
]

models = gpt_4o_mini

work_dir = 'outputs/default/subject'
