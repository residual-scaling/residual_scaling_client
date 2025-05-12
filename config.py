prompt_hyperparameter = {
    "max_summarize_len": 2
}
model_hyperparameter = {
    "temperature": 0.7
}
server_setting = {
    "host": ["127.0.0.1", "127.0.0.1", "127.0.0.1"],
    "port": [4001, 4002, 4003],
    "field": ["law", "science", "biology"],
}
experience_constructor_meta = {
    "max_round": 3
}
model_path = {
    "llama3": "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
}
data_path = {
    "gsm8k": "/root/autodl-tmp/grade-school-math/grade_school_math/data/",
    "sciq": "/root/autodl-tmp/sciq/data/",
    "mmlu": "/root/autodl-tmp/mmlu/",
    # "aqua": "/root/autodl-tmp/aqua.json",
    # "strategyqa": "/root/autodl-tmp/strategyqa.json",
    # "svamp": "/root/autodl-tmp/svamp.json",
    # "multihopqa": "/root/autodl-tmp/multihopqa.json",
    # "mathqa": "/root/autodl-tmp/mathqa.json",
    # "hiddentask": "/root/autodl-tmp/hiddentask.json",
    # "mmlucas": "/root/autodl-tmp/mmlucas.json",
}
