# Training on AWS (GPU)

How to reproduce the training environment on a fresh AWS EC2 GPU instance using the
Deep Learning AMI. Training fits in ~8 GB of VRAM at the default settings.

## 1. Launch the instance

| Setting | Value |
| --- | --- |
| Region | `us-east-1` (N. Virginia) |
| AMI | **Deep Learning OSS Nvidia Driver AMI GPU PyTorch** (Ubuntu) — CUDA + PyTorch preinstalled |
| Instance type | `g4dn.xlarge` (1× T4, 16 GB; cheap) or `g5.xlarge` (1× A10G, 24 GB; faster) |
| Key pair | `deeplearningkey` (reuse the existing one so `deeplearningkey.pem` works) |
| Security group | New group, inbound **SSH (22)** from **My IP** only |
| Storage | AMI default (~225 GiB). Set **Delete on termination = Yes** on *all* volumes |

> **Billing:** a *stopped* instance still bills for its EBS volumes; **terminate** (not stop)
> when finished. The GPU compute is the real cost (`g4dn.xlarge` ≈ $0.53/hr), so terminate
> the moment training is done.

## 2. Connect

From the repo directory on your machine (where `deeplearningkey.pem` lives, `chmod 400`):

```bash
ssh -i deeplearningkey.pem ubuntu@<PUBLIC_IPV4_DNS>
```

The public DNS changes every time the instance is stopped/started, so re-copy it from the
console each session. If SSH times out, your IP likely changed — update the security
group's inbound SSH rule to "My IP".

## 3. Activate the PyTorch virtualenv

This AMI ships PyTorch in a **virtualenv** (not conda — `conda` is not on PATH):

```bash
source /opt/pytorch/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect True
```

## 4. Install dependencies

`torch` (with CUDA), `transformers`, `tokenizers`, `regex`, `safetensors`, and
`huggingface_hub` are **already in the venv**. Do **not** `pip install torch` — it would
clobber the AMI's CUDA build.

Install only what's missing:

```bash
pip install datasets accelerate sacrebleu pynini
```

`accelerate` is required because `common/utils.py::load_model` loads with `device_map=`.

### Match the original library version (recommended)

The released FST model was trained on **transformers 5.5.4** (see `transformers_version`
in `nllb-model-fst-trained/config.json`). To keep the reference model a clean controlled
comparison, pin the same version so a library bump isn't a confounder:

```bash
pip install transformers==5.5.4
```

### `pynini` gotcha

`common/process_word_windows.py` imports `pynini` and reads `analyzeCuzco.windows.fst`
**at import time**, and the training/eval modules import it transitively — so `pynini`
must be installed and able to read that FST file **even for the standard-tokenization
reference run**, which never otherwise touches the FST. Modern `pynini` ships Linux wheels,
so `pip install pynini` usually works without compiling OpenFst. If the FST read fails,
pin `pynini` to a version compatible with the committed `.fst` file.

## 5. Get the code and smoke-test

```bash
git clone https://github.com/MekhiHernandez/quechua-morphoaware-tokenization.git ~/quechua-morphaware-tokenization
cd ~/quechua-morphaware-tokenization
git checkout <branch>

python -c "import datasets, accelerate, sacrebleu, pynini, torch, transformers; print('imports OK', transformers.__version__)"
python -c "from common.process_word_windows import encode_text; print('FST loads OK')"
```

## 6. Train

Download the base model first (the standard reference model needs only this; the FST model
additionally needs `extend_model_vocabulary.py`):

```bash
python initial_load_model.py        # creates ./nllb-model
```

Run training inside **tmux** so it survives SSH disconnects:

```bash
tmux new -s train
python train_reference_model.py 2>&1 | tee train_reference.log
#   detach: Ctrl-b then d   (training keeps running; safe to log out)
#   reattach later: ssh in, then  tmux attach -t train
```

## 7. Retrieve results and tear down

From the repo directory on your machine:

```bash
scp -i deeplearningkey.pem -r \
  ubuntu@<PUBLIC_IPV4_DNS>:~/quechua-morphaware-tokenization/nllb-model-standard-trained/ .
```

Then **terminate** the instance in the EC2 console.
