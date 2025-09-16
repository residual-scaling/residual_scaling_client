prompt_hyperparameter = {
    "max_summarize_len": 3
}
model_hyperparameter = {
    "temperature": 0.7
}
server_setting = {
    "host": ["127.0.0.1", "127.0.0.1", "127.0.0.1"],
    "port": [4001, 4002, 4003],
    "field": ["law", "geography", "biology"],
}
experience_constructor_meta = {
    "max_round": 3
}
model_path = {
    "llama3": "/home/xingkun/nmi-experience_scaling/Meta-Llama-3-8B-Instruct"
}
data_path = {
    "gsm8k": "/home/xingkun/nmi-experience_scaling/grade-school-math/grade_school_math/data/",
    "sciq": "/home/xingkun/nmi-experience_scaling/sciq/data/",
    "mmlu": "/home/xingkun/nmi-experience_scaling/mmlu/",
    # "aqua": "/home/xingkun/nmi-experience_scaling/aqua.json",
    # "strategyqa": "/home/xingkun/nmi-experience_scaling/strategyqa.json",
    # "svamp": "/home/xingkun/nmi-experience_scaling/svamp.json",
    # "multihopqa": "/home/xingkun/nmi-experience_scaling/multihopqa.json",
    # "mathqa": "/home/xingkun/nmi-experience_scaling/mathqa.json",
    # "hiddentask": "/home/xingkun/nmi-experience_scaling/hiddentask.json",
    # "mmlucas": "/home/xingkun/nmi-experience_scaling/mmlucas.json",
}
