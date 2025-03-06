
from rouge import Rouge, FilesRouge
rouge = Rouge()

# the path of the gold
hyp_path = 'summaries_small_gold.txt.tgt'

# the path of the prediction
ref_path = 'summaries_small_pred.txt.tgt'

hypothesis = []
with open(hyp_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        hypothesis.append(l[:-1])

reference = []
with open(ref_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        # if len(l)<3:
            # l = 'ho'
        reference.append(l[:-1])
print(len(reference))

rouge = Rouge()
print("Val", rouge.get_scores(hypothesis, reference, avg = True))
