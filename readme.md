# Tweet sentiment analysis (AI Tinkerer Competition)

### Applied two different approaches

1. "classification" approach
Added a MLP head to T5 model for classification.

Result: **90.7%** accuracy (at step 350)
Weight and Bias training report: https://api.wandb.ai/links/minki-jung/ciihjpfi <br><br><br>
2. "text-to-text" approach
Use text-to-text approach using a prefix. References this paper: https://arxiv.org/pdf/1910.10683v4

Result: **90.662%** accuracy (at step 375)

Weight and Bias training report: https://api.wandb.ai/links/minki-jung/ihrsvww4 <br><br><br>

Dataset used: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment <br><br><br>

note
---------------------
pip install runpod requests datasets transformers numpy evaluate wandb accelerate scikit-learn nltk absl-py rouge-score
