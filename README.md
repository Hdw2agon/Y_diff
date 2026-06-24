# Y-diff: Structure-Texture Decoupled Diffusion Distillation for H&E-to-pCLE Translation
This repository contains the training and evaluation code for a teacher-student diffusion image translation pipeline with latent flow matching. The code supports:

- teacher model training with a DiffAE backbone and optional SPNet mapping module;
- standalone latent Flow Matching (FM) training;
- student model training with the trained teacher and FM module;
- inference and metric evaluation on paired test sets.

## Repository Structure

```text
.
+-- train_diffaeB_map.py          # teacher / student training entry
+-- trainer_fm_transfer.py        # standalone latent Flow Matching training
+-- eval_diffaeB_map.py           # inference and evaluation entry
+-- model/
|   +-- trainer_teacher.py        # teacher trainer, selected by --trainer_version backbone
|   +-- trainer_fm_student.py     # student trainer, selected by --trainer_version fm_student
|   +-- tester_backbone.py        # teacher tester, selected by --tester_version backbone
|   +-- tester_Y_diff.py          # student tester, selected by --tester_version y_diff
|   +-- FlowMatching.py
|   +-- SPNet.py
+-- utils/
|   +-- args.py
|   +-- Metrics.py
|   +-- GAN_loss.py
|   +-- network_ncsn.py
+-- diffae/                       # DiffAE source code / dependencies
+-- CLIP/                         # CLIP source code / dependencies, if used locally
+-- datasets/
```

## Environment

The code is developed for Python and PyTorch with CUDA. A typical setup is:

```bash
conda create -n ydiff python=3.10 -y
conda activate ydiff

# Install PyTorch according to your CUDA version.
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies used by this repository.
pip install numpy pillow tqdm scipy tensorboard wandb opencv-python scikit-image sewar pytorch-fid lpips torchdyn

# OpenAI CLIP package.
pip install git+https://github.com/openai/CLIP.git
```

## Dataset Links

Replace the placeholders with the official download links before release.

| Dataset | Domains | Download |
| --- | --- | --- |
| CCD dataset | `HE -> pCLE` | `TODO: <CCD_DATASET_LINK>` |
| ER-004 dataset | `HE -> IHC` | `TODO: <ER004_DATASET_LINK>` |
| Pretrained DiffAE checkpoint | `ffhq256_autoenc` | `TODO: <DIFFAE_CKPT_LINK>` |

Recommended directory layout:

```text
datasets/
+-- CCD_dataset/
|   +-- HE/
|   |   +-- train/
|   |   +-- test/
|   +-- pCLE/
|       +-- train/
|       +-- test/
+-- ER-004_dataset/
    +-- HE/
    |   +-- train/
    |   +-- test/
    +-- IHC/
        +-- train/
        +-- test/
```

## Checkpoint Preparation

### DiffAE Checkpoint

Both teacher/student training and evaluation load the pretrained DiffAE checkpoint from `--diffae_ckpt`. By default, the code expects:

```text
diffae/checkpoints/
+-- ffhq256_autoenc/
    +-- last.ckpt
    +-- latent.pkl
```

You can also place the checkpoint elsewhere and pass the directory with `--diffae_ckpt`.

### Teacher Checkpoint for Student Training

The student trainer initializes its teacher model from:

```text
prepared_teacher_ckpt/epoch_30.pt
```

After teacher training, copy the selected teacher checkpoint to this path:

```bash
mkdir -p prepared_teacher_ckpt
cp exp/teacher_ccd/ckpt/epoch_30.pt prepared_teacher_ckpt/epoch_30.pt
```

### FM Checkpoint for Student Training and Evaluation

The student trainer and tester load the FM checkpoint from:

```text
fm/<fm_work_dir>/best_fm.pt
```

For the default `--fm_work_dir 2048_8`, the expected path is:

```text
fm/2048_8/best_fm.pt
```

The standalone FM trainer saves checkpoints inside a timestamped run directory, for example `fm/2048_8/best_fm.pt`. Copy or symlink the selected checkpoint to the expected path:

```bash
mkdir -p fm/2048_8
cp fm/2048_8/best_fm.pt
```

## Training

The main training entry is:

```bash
python train_diffaeB_map.py
```

It selects the trainer with `--trainer_version`:

- `backbone`: teacher model, implemented in `model/trainer_teacher.py`;
- `fm_student`: student model, implemented in `model/trainer_fm_student.py`.

### 1. Train the Teacher

Example for the CCD dataset:

```bash
python train_diffaeB_map.py \
  --trainer_version backbone \
  --dataset_path datasets/YOUR_dataset \
  --diffae_ckpt diffae/checkpoints \
  --work_dir exp/teacher_ccd \
  --gpus 0,1 \
  --enable_spnet \
  --n_iter 30 \
  --batch_size 8 \
  --lr 5e-6 \
  --T_train_for 50 \
  --T_train_back 50 \
  --t0_ratio 0.5 \
  --ckpt_freq 5 \
  --print_freq 10
```


Before student training, place the teacher checkpoint at:

```bash
mkdir -p prepared_teacher_ckpt
cp exp/teacher_ccd/ckpt/epoch_30.pt prepared_teacher_ckpt/epoch_30.pt
```

### 2. Train the Latent Flow Matching Module

The FM module learns the latent mapping between source-domain and target-domain DiffAE semantic latents.

```bash
python trainer_fm_transfer.py \
  --dataset_path datasets/YOUR_dataset \
  --diffae_ckpt diffae/checkpoints \
  --work_dir fm \
  --exp_name 2048_8 \
  --batch_size 32 \
  --n_epochs 100 \
  --lr 1e-4 \
  --hidden_dim 2048 \
  --num_layers 8
```

FM checkpoints are saved to:

```text
fm/<exp_name>_<timestamp>/
+-- best_fm.pt
+-- latest_fm.pt
```

Copy the best checkpoint to the path expected by student training and evaluation:

```bash
mkdir -p fm/2048_8
cp fm/2048_8/best_fm.pt
```

### 3. Train the Student

The student trainer uses:

- the pretrained DiffAE checkpoint from `--diffae_ckpt`;
- the teacher checkpoint from `prepared_teacher_ckpt/epoch_30.pt`;
- the FM checkpoint from `fm/<fm_work_dir>/best_fm.pt`.

Example for the CCD dataset:

```bash
python train_diffaeB_map.py \
  --trainer_version fm_student \
  --dataset_path datasets/YOUR_dataset \
  --diffae_ckpt diffae/checkpoints \
  --work_dir exp/student_ccd \
  --gpus 0,1 \
  --enable_spnet \
  --z_sem_flowmatch \
  --fm_work_dir 2048_8 \
  --n_iter 200 \
  --batch_size 8 \
  --lr 5e-6 \
  --T_train_for 50 \
  --T_train_back 50 \
  --t0_ratio 0.5 \
  --ckpt_freq 5 \
  --print_freq 10
```


## Evaluation

The evaluation entry is:

```bash
python eval_diffaeB_map.py
```

It selects the tester with `--tester_version`:

- `backbone`: evaluate a teacher checkpoint with `model/tester_backbone.py`;
- `y_diff`: evaluate a student checkpoint with `model/tester_Y_diff.py`.

### Evaluate the Student

The evaluator loads:

```text
<work_dir>/ckpt/epoch_<n_iter>.pt
```

Therefore, set `--work_dir` to the student experiment directory and set `--n_iter` to the epoch you want to evaluate.

```bash
python eval_diffaeB_map.py \
  --tester_version y_diff \
  --dataset_path datasets/YOUR_dataset \
  --diffae_ckpt diffae/checkpoints \
  --work_dir exp/student \
  --gpus 0 \
  --enable_spnet \
  --z_sem_flowmatch \
  --fm_work_dir 2048_8 \
  --n_iter 200 \
  --batch_size 1 \
  --T_infer_for 50 \
  --T_infer_back 50 \
  --t0_ratio 0.5
```

## Acknowledgements

This code builds on DiffAE, CLIP, LPIPS, PyTorch-FID, and related open-source projects. We thank the authors for their contributions to the community.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{todo2026ydiff,
  title     = {TODO: Paper Title},
  author    = {TODO: Author List},
  booktitle = {TODO: Conference},
  year      = {2026}
}
```
