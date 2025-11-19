# OpenVLA-OFT × LIBERO: No-Training Inference & Evaluation Runbook

This runbook provides a reproducible, minimal path to evaluate OpenVLA-OFT on the LIBERO simulation benchmark without training. It uses:
- Conda env: `libero_openvla` (Python 3.10.14)
- PyTorch 2.2.0 + CUDA 12.1
- Transformers fork: transformers-openvla-oft (auto-installed via project deps)
- In-repo HF cache at `./hf-cache` (as requested)

Primary entrypoint: [eval_libero()](experiments/robot/libero/run_libero_eval.py:461)  
Environment setup script: [scripts/libero_env_setup.sh](scripts/libero_env_setup.sh)  
Smoke test script: [scripts/run_libero_eval_smoke.sh](scripts/run_libero_eval_smoke.sh)

Reference docs:
- Project instructions: [LIBERO.md](LIBERO.md)
- LIBERO env setup helper: [get_libero_env()](experiments/robot/libero/libero_utils.py:18)
- LIBERO extras requirements: [libero_requirements.txt](experiments/robot/libero/libero_requirements.txt:1)

---

## 1) One-time Environment Setup

Use the provided setup script to create and prepare the environment.

- Option A: Default env name (`libero_openvla`)
  ```bash
  bash scripts/libero_env_setup.sh
  ```

- Option B: Custom env name
  ```bash
  bash scripts/libero_env_setup.sh MY_ENV_NAME
  ```

What it does:
- Creates conda env (Python 3.10.14)
- Installs PyTorch 2.2.0 + cu121 wheels
- Editable install of this repo (pulls transformers-openvla-oft per [pyproject.toml](pyproject.toml:47))
- Clones and editable-installs LIBERO into `./LIBERO`
- Installs LIBERO extras from [libero_requirements.txt](experiments/robot/libero/libero_requirements.txt:1)
- Prepares in-repo HF cache at `./hf-cache`

Validation:
- The script prints torch version and CUDA availability.

---

## 2) Headless Rendering and Cache

Before running, set environment variables for headless rendering and in-repo cache. NVIDIA + EGL (preferred):

```bash
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
export HF_HOME="$(pwd)/hf-cache"
export TRANSFORMERS_CACHE="$(pwd)/hf-cache"
```

Notes:
- If EGL is unavailable, try `export MUJOCO_GL=osmesa` (requires `libosmesa6`).
- Some systems need GL libs (e.g., `libgl1`, `libegl1`) installed at OS level.

---

## 3) Run a Smoke-Test Evaluation

Use the convenience script [scripts/run_libero_eval_smoke.sh](scripts/run_libero_eval_smoke.sh). By default it:
- Uses the combined checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`
- Evaluates the `libero_spatial` suite
- Runs `--num_trials_per_task 2` (quick validation)
- Enforces `--center_crop True` (matches training)
- Disables WandB logging

Run:
```bash
bash scripts/run_libero_eval_smoke.sh
```

to run all task specific 
```
conda run -n openvla-oft bash -lc 'export PYTHONPATH="$PWD:$PWD/LIBERO"; export HF_HOME="$PWD/hf-cache"; export TRANSFORMERS_CACHE="$PWD/hf-cache"; python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 --task_suite_name libero_spatial --num_trials_per_task 1 --use_wandb False --center_crop True --onscreen_render True'

```

```
conda run -n openvla-oft bash -lc 'export PYTHONPATH="$PWD:$PWD/LIBERO"; export HF_HOME="$PWD/hf-cache"; export TRANSFORMERS_CACHE="$PWD/hf-cache"; python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task_suite_name libero_spatial --num_trials_per_task 1 --use_wandb False --center_crop True --onscreen_render True' 

```
to use custom instructions 

```
conda run -n openvla-oft bash -lc 'export PYTHONPATH="$PWD:$PWD/LIBERO"; export HF_HOME="$PWD/hf-cache"; export TRANSFORMERS_CACHE="$PWD/hf-cache"; python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 --task_suite_name libero_spatial --num_trials_per_task 1 --use_wandb False --center_crop True --onscreen_render True' --override_task_description "pick up the black bowl next to the ramekin and place it on the plate"
```

to get the task id 

```
conda run -n openvla-oft bash -lc 'python - << "PY"
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
for i in range(suite.n_tasks):
    print(f"{i}: {suite.get_task(i).language}")
PY'
```

To run a particular task with id 

```
conda run -n openvla-oft bash -lc 'export PYTHONPATH="$PWD:$PWD/LIBERO"; export HF_HOME="$PWD/hf-cache"; export TRANSFORMERS_CACHE="$PWD/hf-cache"; python experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 --task_suite_name libero_spatial --single_task_id 3 --num_trials_per_task 1 --use_wandb False --center_crop True --onscreen_render True'
```


Entrypoint implementation: [eval_libero()](experiments/robot/libero/run_libero_eval.py:461)  
Env factory + task description: [get_libero_env()](experiments/robot/libero/libero_utils.py:18)

---

## 4) Outputs and Where to Find Them

- Log file: `./experiments/logs/EVAL-*.txt` (created in [setup_logging()](experiments/robot/libero/run_libero_eval.py:195))
- Episode replay MP4s: `./rollouts/YYYY_MM_DD/*.mp4` (via [save_rollout_video()](experiments/robot/libero/libero_utils.py:47))

The final success rate is printed at the end of [eval_libero()](experiments/robot/libero/run_libero_eval.py:507).

---

## 5) Troubleshooting and Performance Tips

- Center crop:
  - Keep `--center_crop True` because checkpoints were trained with random crop augs ([GenerateConfig.center_crop](experiments/robot/libero/run_libero_eval.py:99) + note in [LIBERO.md](LIBERO.md:82)).

- VRAM / OOM:
  - Add quantization flag for the VLA:
    - `--load_in_8bit True`
    - or `--load_in_4bit True`
  - Flags are consumed by [get_vla()](experiments/robot/openvla_utils.py:253) when creating the HF model.

- Rendering issues:
  - Verify `MUJOCO_GL=egl` and correct `EGL_DEVICE_ID`.
  - If still failing, try `MUJOCO_GL=osmesa` and install OSMesa.  
  - Ensure `robosuite==1.4.1` matches [libero_requirements.txt](experiments/robot/libero/libero_requirements.txt:2).

- Caching:
  - Confirm downloads populate `./hf-cache`.
  - You can clean or relocate cache as needed, but this runbook assumes in-repo cache.

---

## 6) Expanding to Other Suites or Checkpoints

To switch the evaluation:
- Task suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Pretrained checkpoints (suite-specific) from [LIBERO.md](LIBERO.md:44):
  - `moojink/openvla-7b-oft-finetuned-libero-spatial`
  - `moojink/openvla-7b-oft-finetuned-libero-object`
  - `moojink/openvla-7b-oft-finetuned-libero-goal`
  - `moojink/openvla-7b-oft-finetuned-libero-10`
- Combined checkpoint:
  - `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` ([LIBERO.md](LIBERO.md:49))

Example:
```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --num_trials_per_task 5 \
  --use_wandb False \
  --center_crop True
```

---

## 7) Quick Checklist

- [ ] Create env & install: `bash scripts/libero_env_setup.sh`
- [ ] Set env vars: `MUJOCO_GL=egl`, `EGL_DEVICE_ID=0`, `HF_HOME=./hf-cache`, `TRANSFORMERS_CACHE=./hf-cache`
- [ ] Run smoke test: `bash scripts/run_libero_eval_smoke.sh`
- [ ] Inspect logs and videos
- [ ] Expand to other suites or scale trials

If you encounter issues, cross-check defaults and flags in [GenerateConfig](experiments/robot/libero/run_libero_eval.py:82) and utilities in:
- [robot_utils.get_model()](experiments/robot/robot_utils.py:54)
- [openvla_utils.get_vla()](experiments/robot/openvla_utils.py:253)
- [openvla_utils.get_vla_action()](experiments/robot/openvla_utils.py:715)

