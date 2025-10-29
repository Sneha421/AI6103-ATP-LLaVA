## 1.Setup LLaVA 1.5
> This is how you setup the environment
> checkout https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install for more

You will need:
* git (and git-lfs)
* conda with python 3.10

### 1.1 Get the code & weights
1. clone LLaVA repo
```bash
# clone LLavA repo
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```
2. clone LLaVA weights
```bash
# setup git-lfs
git lfs install

# clone weights
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

# alternatively, use huggingface CLI (for missing git-lfs like NSCC envs)
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir ./llava-v1.5-7b
```

3. setup conda & dependencies
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### 1.2 Evaluate without Pruner
> checkout https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md for more

```bash
# download data
curl -L -o ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv

# evaluate
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh
```
