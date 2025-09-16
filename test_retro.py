import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from util import log_flush
from model.model_loader import load_model
import pandas as pd
from experiment_analyzer.retrospect_analyzer import RetrospectAnalyzer
from retro_tester import RetroTester
from data_loader.loader_util import get_dataloader

model_path = "/home/xingkun/nmi-experience_scaling/Meta-Llama-3-8B-Instruct"
model = load_model("local", model_path)

# hyper parameter
subject = "biology"
dataset="MMLU"
subset = "high_school_biology"
round_amount = 50000

ds = get_dataloader(dataset, subset)

ds_size = ds.get_size()
print(ds.get_size())
test_amount=ds_size 

analyzer = RetrospectAnalyzer(ds.get_size())

rt = RetroTester(
    id="01",
    subject=subject,
    dataset=dataset,
    subset=subset,
    test_amount=test_amount,
    round_amount=10,
    model=model,
    analyzer=analyzer,
    use_experience=True,
    trace_scores=True,
    save_experience=True,
    save_distill_experience=True
)

for i in range(round_amount):
    print(i)
    rt.start(test_amount)
    rt.teacher_model.save_experience_db()

len(rt.dataset)
