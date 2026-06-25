# Joint Denoising：视频 + 触觉联合去噪（TACO 集成）

参考实现：`DiffSynth-Action reference implementation`
（基于 DiffSynth-Studio + Wan2.2-TI2V-5B 的 joint denoise 流水线）

本文记录该方案在 **TACO** 框架下的对应实现位置、改动内容、与参考实现的差异，以及如何启动训练 / 推理。

---

## 1. 方法总览

| 模式 | 视频 | 触觉/动作 | 训练目标 |
|------|------|-----------|----------|
| **Joint Denoising（本方案）** | 去噪 | 同步加噪 + 去噪 | 视频 + 触觉 flow-matching loss |

DiT 自注意序列被扩展为 `[ video tokens (N_vid) | tactile tokens (T) ]`，
视频/触觉 token 在 self-attention 内完全双向交互；DiT 输出后再按 `N_vid` 切片：
前半走原 unpatchify 得到视频速度场，后半经 `WanTactileHead` 得到触觉速度场。

触觉默认维度 `tactile_dim = 12`：左/右手力传感器各 6 维（fx,fy,fz,tx,ty,tz）。当前实现推荐使用**按通道 z-score 归一化**（`(x - mean) / std`），让触觉分支进入与高斯噪声同阶的数值空间；若不提供 `tactile_stats_path`，则回退为原始物理量直训。

---

## 2. 文件改动清单

### 新增文件

| 文件 | 内容 |
|------|------|
| `visuo_tactile_world_model/world_model/model/wan/wan_video_tactile.py` | `WanTactileTokenizer` + `WanTactileHead`。Tokenizer 注入 timestep 的 sinusoidal embedding **+ 1D sinusoidal 位置编码（沿 T 轴）**，`out_proj` zero-init；Head 把 DiT 输出投回 `(B,T,12)`。 |
| `visuo_tactile_world_model/world_model/model/wan/pipeline_ti2v_5b_joint_denoise.py` | `WanTI2V5BJointDenoisePipeline`（继承 `WanTI2V5BPipeline`），`from_pretrained` 时按 `dit.dim/freq_dim` 实例化 tactile 模块；`model_fn_wan_video_joint_denoise` 处理 token 拼接、**RoPE freqs（tactile 时间维映射到 video latent f 轴；h/w 用单位复数 `1+0j`）**、**`tactile_anchor_first` 控制下的 t_mod 扩展（首帧 anchor 用 timestep=0，其余用当前 timestep）**、双输出切分。`__call__` 内置首帧 anchor 与每步 overwrite。`in_iteration_models = ("dit","tactile_tokenizer","tactile_head")`。 |
| `examples/CustomDataset/train_worldrl_stride5_192x256_joint_ti2v_5b.yaml` | WorldRL 联合去噪训练配置（**当前指向 base pipeline / `sft:train`，需改为 joint，详见 §4 配置校对**）。 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `visuo_tactile_world_model/world_model/dataset/operators.py` | 新增 `LoadNumpyArray`：读 `.npy`，按 `num_frames` 用 `np.linspace` 做与视频帧一致的时序降采样，输出 `torch.float32` 张量。 |
| `visuo_tactile_world_model/world_model/runner/runner_util/wan_runtime.py` | `WAN_I2V_DEFAULTS` 增加 `tactile_dim/tactile_loss_weight/tactile_hidden_dim/tactile_num_layers/tactile_sequence_key`；`_coerce_runtime_types` 把这些字段加入 int/float 强转集合；`build_wan_training_dataset` 的 `special_operator_map` 中为 `force_sequence` / `tactile_sequence` 三个键挂上 `LoadNumpyArray(num_frames=args.num_frames)`。 |
| `visuo_tactile_world_model/world_model/model/loss.py` | 新增 `FlowMatchJointDenoiseLoss`：视频和触觉共用同一 `timestep`，分别 `add_noise` + `training_target`，DiT 一次前向同时产出两路速度场，损失为 `loss_video + tactile_loss_weight * loss_tactile`（再乘 `pipe.scheduler.training_weight(timestep)`）。**首帧 anchor**：`noisy_tactile[:,0:1]` 被 overwrite 回 clean GT，loss 排除该位置（`[:, 1:]`），并把 `tactile_anchor_first=True` 透传给 `model_fn`，与视频侧 `first_frame_latents` 的处理对称。 |
| `visuo_tactile_world_model/world_model/runner/wan/wan_training.py` | `WanTrainingModule.task_to_loss` 注册 `sft_joint_denoise`、`sft_joint_denoise:train`、`sft_joint_denoise:data_process`；`parse_extra_inputs` 接受 `tactile_sequence`/`force_sequence`，2D `(T,12)` 自动扩成 3D `(1,T,12)` 后写入 `inputs_shared["tactile_input"]`；`get_pipeline_inputs` 注入 `tactile_loss_weight`；`forward` 缓存路径 `setdefault("tactile_loss_weight", ...)`。`WanTrainRunner` / `WanCacheRunner` 把 `tactile_loss_weight`、`tactile_sequence_key` 从 args 透传给 `WanTrainingModule`。 |
| `visuo_tactile_world_model/world_model/runner/wan/wan_data_preprocess.py` | 同上 cache runner 透传。 |

---

## 2.5 各 Block 详细结构

### 2.5.1 WanTactileTokenizer

将带噪触觉序列 `(B, T, 12)` + 当前 timestep 编码为 DiT 兼容的 token 序列 `(B, T, 5120)`。

```
输入: x (B, T, D_tac=12),  timestep (1,) or (B,)

feat_proj:
  Linear(D_tac=12 → H=512) + SiLU
  → (B, T, H)

time_proj:
  sinusoidal_embedding_1d(freq_dim=256, timestep)   # → (1, 256) or (B, 256)
  → Linear(256 → H) + SiLU + Linear(H → H)
  → t_emb: (B, H) → unsqueeze(1) → (B, 1, H)

pos_emb:
  sinusoidal_embedding_1d(H, arange(T))             # 1D 时序 PE，沿 T 轴
  → (T, H) → unsqueeze(0) → (1, T, H)

h = feat_proj(x) + pos_emb + t_emb                 # → (B, T, H)

transformer:
  2 × TransformerEncoderLayer(
      d_model=H=512, nhead=8,
      dim_feedforward=H×4=2048,
      norm_first=True,            # Pre-LN
      batch_first=True,
  )
  → (B, T, H)

out_proj (zero-init):
  Linear(H=512 → model_dim=5120)
  → (B, T, 5120)
```

关键设计：
- `feat_proj` zero-init 不是 `out_proj` 的替代，是 `out_proj` zero-init——保证训练初始触觉 token 对 DiT 贡献为零，避免破坏预训练视频权重。
- `time_proj` 双层 MLP 对 sinusoidal 嵌入做非线性变换，使 timestep 信息可被各 T 位置共享（广播加法）。
- `pos_emb` 独立于 timestep，让 Transformer 感知触觉的时序顺序（否则 self-attn 对 T 置换等变）。

---

### 2.5.2 WanTactileHead

将 DiT 输出的触觉 token 序列 `(B, T, 5120)` 解码回速度场 `(B, T, 12)`。

```
输入: x (B, T, model_dim=5120)

head:
  LayerNorm(5120)
  Linear(5120 → H=512) + SiLU
  Linear(H=512 → D_tac=12)

输出: (B, T, 12)    ← 触觉速度场预测 v̂_tac
```

> 轻量设计：总参数 ≈ 5120×512 + 512×12 ≈ 2.6M，相比 DiT 的 5B 参数可忽略。

---

### 2.5.3 model_fn_wan_video_joint_denoise（联合前向）

一次前向同时输出视频速度场和触觉速度场。

```
输入:
  dit, tactile_tokenizer, tactile_head
  latents       (B, C, F, H, W)   — 带噪视频 latent
  noisy_tactile (B, T, 12)        — 带噪触觉（第0帧已被 clean overwrite）
  timestep      (1,) or (B,)
  context       (B, L_text, dim)  — T5 文本 embedding
  clip_feature  (B, L_clip, ...)  — CLIP image embedding（可选）
  y             (B, C, F, H, W)   — VAE embedding（可选，image cond）
  tactile_anchor_first: bool

① timestep embedding
   sinusoidal_1d(freq_dim, t) → time_embedding → time_projection
   t_mod: (B, N_tok, 6, dim)  [per-token 模式] 或 (B, 6, dim)

② text embedding
   text_embedding(context) → (B, L_text, dim)

③ video patchify
   latents → patchify(x) → video_tokens: (B, N_vid, dim)
   N_vid = f·h·w  [latent 空间分辨率]

④ tactile tokenize
   tactile_tokenizer(noisy_tactile, timestep) → tactile_tokens: (B, T_tac, dim)

⑤ sequence concat
   x = cat([video_tokens, tactile_tokens], dim=1)  → (B, N_vid+T_tac, dim)

⑥ t_mod 扩展（per-token 模式时）
   为 T_tac 个 tactile 位置追加 t_mod：
   - tactile_anchor_first=True : tac_ts = [0, t, t, ..., t]   # 首帧用 t=0
   - tactile_anchor_first=False: tac_ts = [t, t, ..., t]
   t_mod = cat([t_mod_video, t_mod_tac], dim=1)

⑦ RoPE freqs 扩展
   video freqs : (N_vid, 1, dim_rope)
     = cat(freqs_t[f], freqs_h[h], freqs_w[w]).reshape(f·h·w, ...)

   tactile freqs: (T_tac, 1, dim_rope)
     t 轴: linspace(0, f-1, T_tac).round().long() → 取 dit.freqs[0] 对应行
           （tactile[i] 对齐到 video frame round(i·(f-1)/(T_tac-1))）
     h 轴: ones(T_tac, dim_h) × (1+0j)   # 单位复数，无旋转
     w 轴: ones(T_tac, dim_w) × (1+0j)

   freqs = cat([video_freqs, tac_freqs], dim=0)

⑧ DiT blocks（40 层 WanAttentionBlock）
   for block in dit.blocks:
       x = block(x, context, t_mod, freqs)
   → (B, N_vid+T_tac, dim)
   video / tactile token 在每层 self-attention 内完全双向交互

⑨ 序列切分
   video_tokens_out    = x[:, :N_vid]         # (B, N_vid, dim)
   tactile_tokens_out  = x[:, N_vid:]         # (B, T_tac, dim)

⑩ video 解码
   dit.head(video_tokens_out, t) → unpatchify → video_pred: (B, C, F, H_px, W_px)

⑪ tactile 解码
   tactile_head(tactile_tokens_out) → tactile_pred: (B, T_tac, 12)

输出: (video_pred, tactile_pred)   [has_tactile=True]
      video_pred                   [has_tactile=False]
```

---

### 2.5.4 FlowMatchJointDenoiseLoss

视频和触觉共用同一个随机 timestep，一次 DiT 前向产出两路速度场，联合计算 loss。

```
① 采样 timestep
   t_id ~ Uniform[t_min_id, t_max_id)
   t = scheduler.timesteps[t_id]              # scalar

② video 加噪
   ε_vid ~ N(0, I)
   noisy_latent = add_noise(clean_latent, ε_vid, t)
   target_vid   = training_target(clean_latent, ε_vid, t)  # flow-matching velocity
   noisy_latent[:, :, 0:1] = first_frame_latents           # image-cond clean anchor

③ tactile 加噪
   ε_tac ~ N(0, I)
   noisy_tac  = add_noise(tactile_clean, ε_tac, t)
   target_tac = training_target(tactile_clean, ε_tac, t)
   noisy_tac[:, 0:1] = tactile_clean[:, 0:1]              # force_0 clean anchor

④ 联合前向
   (noise_pred_vid, noise_pred_tac) = model_fn(
       latents=noisy_latent,
       noisy_tactile=noisy_tac,
       tactile_anchor_first=True,
       timestep=t, ...
   )

⑤ video loss（排除第0帧 image-cond）
   loss_vid = MSE(noise_pred_vid[:, :, 1:], target_vid[:, :, 1:])

⑥ tactile loss（排除第0帧 force anchor）
   loss_tac = MSE(noise_pred_tac[:, 1:], target_tac[:, 1:])

⑦ 合并
   w = training_weight(t)           # scheduler 对当前 timestep 的加权
   loss = (loss_vid + λ · loss_tac) · w
          ↑ λ = tactile_loss_weight，默认 0.1
```

对称性说明：

| 维度 | 视频 | 触觉 |
|------|------|------|
| anchor 位置 | frame 0（VAE clean image） | frame 0（force_0 clean） |
| noisy 输入 | `latents[:,:,0:1] = clean` | `noisy_tac[:,0:1] = clean` |
| loss 排除 | `[:,:,1:]` | `[:,1:]` |
| `model_fn` 控制 | `first_frame_latents` | `tactile_anchor_first=True` |
| t_mod 修改 | `t_mod[0]` 用 `t=0`（`seperated_timestep`） | `tac_ts[0] = 0` |

---

## 3. 数据流

### 3.1 数据预处理（`task: sft_joint_denoise:data_process`）

1. `UnifiedDataset` 按 metadata CSV 列读样本，`force_sequence` / `tactile_sequence` 列由 `LoadNumpyArray` 读取并对齐到 `args.num_frames`（与视频帧 `np.linspace` 同步）。
2. `WanTrainingModule.parse_extra_inputs` 将其装入 `inputs_shared["tactile_input"]`，并随 `input_latents`、`first_frame_latents`、`prompt_emb` 等一起经 `pipe.units` 走完编码 unit。
3. `:data_process` 走 identity loss，把 `(inputs_shared, inputs_posi, inputs_nega)` 元组 `torch.save` 到 `output_path/<rank>/<idx>.pth`。

> 当前训练 YAML 指向的 cache 目录已存在大量 `.pth`：
> `/path/to/worldrl/wan_worldrl_stride5_192x256/latent_cache_ti2v_5b_joint_nf37_s1/0/*.pth`
> 若该批 cache 是用**老的 `sft:data_process`** 跑出来的，则其中**没有** `tactile_input`，需要用 `sft_joint_denoise:data_process` 重新生成 cache。

### 3.2 训练（`task: sft_joint_denoise:train`）

1. `UnifiedDataset` 直接 `torch.load` 缓存 pth → 得到三元组 `sample`。
2. `WanTrainingModule.forward(data={}, inputs=sample)`：跳过 `pipe.units`（latents 已缓存），调用 `task_to_loss["sft_joint_denoise:train"]` → `FlowMatchJointDenoiseLoss`。
3. Loss 内：
   - 同一个随机 `timestep` 同时给视频 latent / 触觉序列加噪；
   - `pipe.model_fn = model_fn_wan_video_joint_denoise` 拼接 token 并返回 `(noise_pred_video, noise_pred_tactile)`；
   - 总 loss = `loss_video + tactile_loss_weight * loss_tactile`，再乘 `scheduler.training_weight(t)`。

### 3.3 触觉归一化（当前实现）

当前仓库里的 tactile 归一化已经落地，行为如下：

1. **cache 阶段不归一化**：`sft_joint_denoise:data_process` 写入 `.pth` 时保留原始 tactile/force 数值，便于后续切换不同统计量重新训练，而不用重跑 cache。
2. **训练阶段按通道 z-score**：若 yaml 里提供 `tactile_stats_path`，`WanTrainingModule.forward` 会在进入 `FlowMatchJointDenoiseLoss` 前把 `tactile_input` 变成 `(x - mean) / std`。
3. **验证阶段用同一份 stats**：`TactileVideoInferenceDataset(stats_path=...)` 会对 `tactile_init` 和 `tactile_gt` 做同样的 z-score，保证 train / val 处在同一个空间。
4. **自定义推理脚本也要保持一致**：如果模型训练时用了 `tactile_stats_path`，那么脚本传入 `tactile_init` 前也必须先按同一份 `mean/std` 归一化；若需要导出物理量，再把模型输出做反归一化。

推荐先用仓库里的 `scripts/compute_force_stats.py` 基于训练 metadata 计算 `mean/std`，再把生成的 `.npz` 同时填到训练 yaml 的 `runner.params.tactile_stats_path` 和验证集的 `validation_dataset.params.stats_path`。

---

## 4. 配置校对（重要）

`examples/CustomDataset/train_worldrl_stride5_192x256_joint_ti2v_5b.yaml` **当前**仍指向基础 SFT，**未真正启用 joint denoise**。要让上面提到的代码路径生效，必须改成下面这样：

```yaml
runner:
  class_path: visuo_tactile_world_model.world_model.runner.wan.wan_training.WanTrainRunner
  params:
    # ---- 之前正确的字段保持不变 ----
    data_file_keys: image,video,force_sequence  # 必须显式声明，否则 UnifiedDataset 不会调用 operator，cache 里就没有 force tensor
    extra_inputs: input_image,force_sequence    # 加入 force_sequence (或 tactile_sequence)
    trainable_models: dit,tactile_tokenizer,tactile_head   # tactile 模块要参与训练
    remove_prefix_in_ckpt: pipe.                # 让 dit + tactile_* 都被保存
    task: sft_joint_denoise:train               # 启用 joint denoise loss
    tactile_loss_weight: 0.1                    # 可选，默认 0.1
    tactile_dim: 12                             # 可选，默认 12
    tactile_hidden_dim: 512
    tactile_num_layers: 2

model:
  class_path: visuo_tactile_world_model.world_model.model.wan.pipeline_ti2v_5b_joint_denoise.WanTI2V5BJointDenoisePipeline
  params:
    device: cuda
    torch_dtype: bfloat16
    tokenizer_path: checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl
    model_paths:
      - - checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
        - checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
        - checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
      - checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
      - checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

校对要点：

1. **pipeline class 必须是 joint 版本**，否则 `pipe.tactile_tokenizer/head` 不会被实例化，loss 里的 `noisy_tactile` 走不到 `model_fn` 的 tactile 分支，整个 joint 路径形同虚设。
2. **task 必须是 `sft_joint_denoise:train`**（cache 阶段同理用 `sft_joint_denoise:data_process`），否则 `task_to_loss` 仍走 `FlowMatchSFTLoss`。
3. **`extra_inputs` 必须包含触觉键**，否则 cache 阶段 `inputs_shared["tactile_input"]` 不会被写入 pth，训练阶段 `FlowMatchJointDenoiseLoss` 看到 `tactile_input is None`，会 silent 退化为纯视频 loss。
4. **`trainable_models` 要包含 `tactile_tokenizer,tactile_head`**。这两个模块是从随机/zero-init 开始训练的；只训 `dit` 它们的权重永远不动，触觉预测 = 常数，loss 不会下降。
5. **`remove_prefix_in_ckpt: pipe.`**（注意只有 `pipe.`，不是 `pipe.dit.`）。否则保存的 state_dict 会丢掉 `tactile_tokenizer.*` 与 `tactile_head.*`。
6. **数据列**：`metadata_taco_joint.csv` 必须包含 `force_sequence`（或 `tactile_sequence`）字段，值是相对 `dataset_base_path` 的 `.npy` 路径，shape 至少 `(T_raw, 12)`，`LoadNumpyArray` 会按 `num_frames` 重采样。
7. **`data_file_keys` 必须显式包含触觉列**：`UnifiedDataset.__getitem__` 只对 `data_file_keys` 里列出的 key 调用 operator；`special_operator_map` 注册了 `force_sequence` 也没用，必须在 yaml 写 `data_file_keys: image,video,force_sequence`。漏了这一行 → CSV 字符串路径原样进 `parse_extra_inputs` → `.ndim` 立刻报 AttributeError。
8. **cache 复用**：当前 `latent_cache_ti2v_5b_joint_nf37_s1/` 下已经有缓存 pth。若它们是用 joint 任务生成的，`tactile_input` 已写入，可以直接训练；若是老 cache，请用上面 §3.1 的 data_process 任务重新生成（或在 metadata CSV 上重跑）。

---

## 5. SFT 数据 Tail Window 覆盖

### 问题

用固定步长（stride）生成训练窗口时，若视频长度 N 不能被 `stride` 整除，则最后一段 `[N - window_size, N]` 不会出现在任何训练样本中。推理时以 `N - window_size` 帧作为 conditioning 输入，模型从未见过这类起始位置，对该阶段的视频和触觉预测都会退化。

### 解决方式

在 cache 构建 yaml 中设置（joint cache yaml 已默认包含该选项）：

```yaml
context_window_tail_align: true
```

开启后，对每个视频（共 N 帧）：

- stride 窗口覆盖起点：`0, stride, 2·stride, …, last_covered`（窗口完整落在视频内）
- 额外追加起点：`last_covered+1, last_covered+2, …, N-1`，每个窗口的视频 latent 和 `tactile_input` 均用 **最后一帧补满 window_size**（`pad_last=True`）

效果：**原视频每一帧都作为训练起始帧**，tail 段的视频与触觉序列一同被补全。需要重新跑 cache 才能生效。

---

## 6. 启动命令

### 重新生成 joint cache（仅当现有 cache 没有 `tactile_input` 时需要）

仓库实际的缓存脚本是 **`scripts/process_cache.sh`**（不是 `data_process.sh`），并且 cache / train 用的是**两份独立的 yaml**（参见 `examples/CustomDataset/cache_ti2v_5b.yaml`）。所以你需要先复制一份 cache yaml，再把它改成 joint denoise 形态，例如 `examples/CustomDataset/cache_worldrl_stride5_192x256_joint_ti2v_5b.yaml`：

```yaml
task: cache

runner:
  class_path: visuo_tactile_world_model.world_model.runner.wan.wan_data_preprocess.WanCacheRunner
  params:
    output_path: ./data/your_dataset/latent_cache_ti2v_5b_joint   # 会被 --output-path 覆盖
    height: 192
    width: 256
    num_frames: 37
    fps: 16
    resize_mode: letterbox
    context_window_short_video_mode: drop
    context_window_stride: 5
    context_window_tail_align: true
    context_window_wait_timeout: 7200
    data_file_keys: image,video,force_sequence            # ← 关键：UnifiedDataset 只对这里列出的 key 跑 operator，少这一行 force_sequence 不会被读成 tensor
    extra_inputs: input_image,force_sequence              # ← 关键，把触觉拉进 cache
    trainable_models: dit,tactile_tokenizer,tactile_head
    task: sft_joint_denoise:data_process                  # ← 关键
    tactile_dim: 12

model:
  class_path: visuo_tactile_world_model.world_model.model.wan.pipeline_ti2v_5b_joint_denoise.WanTI2V5BJointDenoisePipeline
  params:
    device: cuda
    torch_dtype: bfloat16
    tokenizer_path: checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl
    model_paths:
      - - checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
        - checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
        - checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
      - checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
      - checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

dataset:
  class_path: visuo_tactile_world_model.world_model.dataset.unified_dataset.UnifiedDataset
  params:
    dataset_base_path: /path/to/worldrl/wan_worldrl_stride5_192x256
    dataset_metadata_path: /path/to/worldrl/wan_worldrl_stride5_192x256/metadata_taco_joint.csv

runtime:
  params:
    gradient_accumulation_steps: 1
```

跑 cache：

```bash
bash scripts/process_cache.sh \
  --config       examples/CustomDataset/cache_worldrl_stride5_192x256_joint_ti2v_5b.yaml \
  --dataset-base-path /path/to/worldrl/wan_worldrl_stride5_192x256 \
  --metadata-path     /path/to/worldrl/wan_worldrl_stride5_192x256/metadata_taco_joint.csv \
  --output-path       /path/to/worldrl/wan_worldrl_stride5_192x256/latent_cache_ti2v_5b_joint_nf37_s1 \
  --overwrite
```

> `process_cache.sh` 自动 override `dataset_base_path` / `dataset_metadata_path` / `output_path`，但**不**覆盖 `task` / `extra_inputs` 等 runner 字段，所以这些必须事先写在 yaml 里。

### 训练

```bash
cd /path/to/TACO
conda activate visuo_tactile_world_model.world_model_backup

bash scripts/train_full.sh \
  --config examples/CustomDataset/train_worldrl_stride5_192x256_joint_ti2v_5b.yaml \
  --dataset-base-path /path/to/worldrl/wan_worldrl_stride5_192x256/latent_cache_ti2v_5b_joint_nf37_s1
```

> 单卡 debug 已在 `.vscode/launch.json` 中配好 `Debug: train_full worldrl (single GPU, visuo_tactile_world_model.world_model_backup)`：该项已显式绑定 `visuo_tactile_world_model.world_model_backup` 的 Python 解释器，并以 `CUDA_VISIBLE_DEVICES=0` 直接 `debugpy` 启动 `run.py`，断点可命中训练循环。

---

## 6. 与参考实现（DiffSynth-Action）的差异

| 维度 | 参考实现 | TACO |
|------|----------|----------|
| 触觉模块 | `diffsynth/models/wan_video_tactile_joint_denoise.py` | `visuo_tactile_world_model/world_model/model/wan/wan_video_tactile.py`（结构一致：feat_proj + time_proj + 2 层 TransformerEncoder + zero-init out_proj） |
| 联合 model_fn | `examples/wanvideo/model_training/train_joint_denoise.py: model_fn_wan_video_joint_denoise` | `visuo_tactile_world_model/world_model/model/wan/pipeline_ti2v_5b_joint_denoise.py: model_fn_wan_video_joint_denoise`（语义一致） |
| Loss | `FlowMatchJointDenoiseLoss` 返回 `(total, video, tactile)` 三元组用于 wandb | TACO 直接返回标量 total（wandb 只记 `train/loss`，未拆分两路） |
| 数据集 | `SlidingWindowEpisodeDataset`（按 episode 目录滑窗） | 复用 `UnifiedDataset` + 预先生成的 metadata CSV / 缓存 pth；`ContextWindow` 机制由 `build_context_window_metadata` 完成等价滑窗展开 |
| 推理 | `WanTactileJointDenoisePipeline.__call__` 内置 joint 采样、`return_action_tactile=True` | **已实现 + 增强**：`WanTI2V5BJointDenoisePipeline.__call__` 重写了 base 采样循环，每步同时给 tactile 加噪并由 `model_fn` 输出双速度场，scheduler 同步 step。CFG 仅作用于视频；tactile 用 positive 预测。**首帧 anchor**：`tactile_init` 接受完整 GT 或仅 force_0，第 0 帧每步强制 overwrite 为 clean，配合 model_fn 的 `tactile_anchor_first=True` 把第 0 帧 modulation 设为 timestep=0，物理上对应"image + force_0 → video + force"。若训练使用了 `tactile_stats_path`，则 `tactile_init` 应处于同一 z-score 空间，模型输出的 tactile 也处于该归一化空间；需要物理量时再反归一化。开关：`return_tactile: bool = False`（默认 False 保证 validation pipeline 不破坏；自定义脚本里设 True 拿到 `(video, tactile)`）。 |
| wandb 指标 | `loss/total`、`loss/video`、`loss/tactile` 分别记录 | 暂只记 `train/loss`（如需拆分需改 `FlowMatchJointDenoiseLoss` 返回结构 + `launch_training_task` 日志逻辑） |

---

## 7. 完成度自检 Checklist

代码层（已完成）：
- [x] WanTactileTokenizer / WanTactileHead 模块
- [x] joint model_fn（token 拼接、freqs padding、t_mod 扩展、双输出切分）
- [x] WanTI2V5BJointDenoisePipeline 及 `from_pretrained` 自动建模
- [x] FlowMatchJointDenoiseLoss
- [x] `sft_joint_denoise(:train|:data_process)` 任务路由
- [x] `LoadNumpyArray` 操作符 + `force_sequence/tactile_sequence` 列绑定
- [x] tactile 相关 args 默认值与类型强转
- [x] `parse_extra_inputs` 支持触觉键
- [x] `context_window_tail_align: true`：追加 `[last_covered+1, N-1]` 所有起点（`pad_last=True`），确保每帧都作为训练起始帧，视频与触觉一同补全
- [x] `WanInferRunner` 支持 `"tail"` fraction 精确命中每个视频的 `N - window_size` 帧
- [x] `WanInferRunner` 推理时自动生成 GT vs 生成对比视频 `*_compare.mp4`

配置/数据层（仍需手工确认）：
- [ ] `train_worldrl_stride5_192x256_joint_ti2v_5b.yaml` 切换到 joint pipeline class + `sft_joint_denoise:train` + 触觉 extra_input + tactile 模块加入 `trainable_models` + `remove_prefix_in_ckpt: pipe.`（见 §4）
- [ ] metadata CSV 包含 `force_sequence` 列、对应 `.npy` 已生成
- [ ] 现有 `latent_cache_ti2v_5b_joint_nf37_s1` 是否含 `tactile_input`（需要 joint data_process 重跑或确认）

可选增强（参考实现已有，本仓库尚缺）：
- [x] 推理侧 joint 采样：`WanTI2V5BJointDenoisePipeline.__call__` 已重写，传 `return_tactile=True` 即返回 `(video, tactile_pred)`，shape `(1, num_frames, tactile_dim)`；若训练启用了 `tactile_stats_path`，其数值域为 z-score 归一化空间。
- [ ] wandb 拆分 `loss/video` 与 `loss/tactile`
- [x] 触觉按通道 z-score 归一化（train/val 共用同一份 `mean/std`）
- [ ] `tanh` 软饱和（可选增强），用于 wipe_whiteboard 类大扭矩长尾场景

---

## 8. 推理：从噪声同时生成视频 + 触觉

```python
import torch, numpy as np
from PIL import Image
from visuo_tactile_world_model.world_model.model.wan.pipeline_ti2v_5b_joint_denoise import (
    WanTI2V5BJointDenoisePipeline,
)
from visuo_tactile_world_model.world_model.utils.loader.config import ModelConfig

pipe = WanTI2V5BJointDenoisePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # DiT 三个 shard 必须合到同一个 ModelConfig，否则会被当作三个独立模型分别加载、报 missing key
        ModelConfig(path=[
            "checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
            "checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
            "checkpoints/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
        ]),
        ModelConfig(path="checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    ],
    tactile_dim=12,
)

# 加载训练得到的 tactile_tokenizer / tactile_head（以及可选的 dit）
# checkpoint 由 ModelLogger 以 safe_serialization=True 写入，必须用 safetensors loader
from safetensors.torch import load_file

sd = load_file("logs/train/.../step-XXXX.safetensors")

# YAML 里 remove_prefix_in_ckpt: pipe.   →   存盘时已剥掉 "pipe." 前缀，
# 所以这里的 key 形如  tactile_tokenizer.feat_proj.0.weight
tok_sd  = {k[len("tactile_tokenizer."):]: v
           for k, v in sd.items() if k.startswith("tactile_tokenizer.")}
head_sd = {k[len("tactile_head."):]: v
           for k, v in sd.items() if k.startswith("tactile_head.")}
pipe.tactile_tokenizer.load_state_dict(tok_sd, strict=True)
pipe.tactile_head.load_state_dict(head_sd, strict=True)
pipe.tactile_tokenizer = pipe.tactile_tokenizer.to(torch.bfloat16).cuda()
pipe.tactile_head      = pipe.tactile_head.to(torch.bfloat16).cuda()

# 如果你也存了 dit 的全量 / lora 权重，同样按前缀切分后 load_state_dict 进 pipe.dit。

input_image = Image.open("data/pick_flower/0/frame0.jpg")

stats = np.load("/path/to/worldrl/wan_worldrl_stride5_192x256/force_stats.npz")
force_mean = torch.from_numpy(stats["mean"]).float()
force_std = torch.from_numpy(stats["std"]).float().clamp_min(1e-6)

force_0_raw = torch.tensor([...], dtype=torch.float32)   # shape (12,)，原始物理量
force_0 = (force_0_raw - force_mean) / force_std         # 若训练用了 tactile_stats_path，这里必须先做同样的 z-score

video, tactile = pipe(
    prompt="robot picks a flower from the pot",
    negative_prompt="低质量，模糊，静态",
    input_image=input_image,
    height=192, width=256, num_frames=37,
    num_inference_steps=25,
    cfg_scale=1.0,
    seed=42,
    tiled=True,
    tactile_init=force_0,          # ← 首帧 anchor（仅 force_0）；也可传完整 (N,12) GT
    return_tactile=True,           # ← 关键
)
# tactile: torch.Tensor float32, shape (1, num_frames, tactile_dim)
# 注意：若训练用了 tactile_stats_path，这里拿到的是“归一化空间”预测。
tactile_raw = tactile * force_std.view(1, 1, -1) + force_mean.view(1, 1, -1)
np.save("output_tactile.npy", tactile_raw.numpy())
```

参数说明：

| 参数 | 含义 |
|------|------|
| `return_tactile=True` | 返回 `(video, tactile)`；默认 `False` 兼容现有 validation 调用。若训练用了 `tactile_stats_path`，返回的 `tactile` 位于归一化空间，需要调用方自行反归一化成物理量。 |
| `tactile_init` | **首帧 anchor 入口**。两种用法：① **仅 force_0**（部署常用）—— shape `(D,)` / `(1, D)` / `(1, 1, D)`，1..N-1 帧用纯噪声，第 0 帧放 clean GT；② **完整 GT**（验证/可视化）—— shape `(N, D)` / `(1, N, D)`，整段加噪到 `timesteps[0]`，第 0 帧 overwrite 回 clean。两种路径都会启用 `tactile_anchor_first=True` 并在每个 scheduler step 后强制 `tactile[:, 0:1] = anchor`，与训练侧一致。若训练用了 `tactile_stats_path`，传入这里的值也必须已经过同一份 `mean/std` 归一化。`None` 则纯噪声、无 anchor（与训练分布不一致，仅 fallback）。 |
| CFG | 仅作用于视频分支；tactile 始终走 positive prediction（FAQ 所述）。 |

---

## 9. 设计修订：位置编码 + RoPE + 首帧 anchor

相对参考实现，本仓库针对 tactile 的几个语义/数值问题做了如下修订：

### 9.1 tactile token 的 1D 位置编码（tokenizer 内）

**问题**：`WanTactileTokenizer` 用普通 `TransformerEncoderLayer`，原始实现只加 timestep embedding，没有沿 T 轴的 PE → 对 T 置换等变，时间顺序信息丢失。

**修复**：`forward` 里在 `feat_proj(x)` 之后加上 `sinusoidal_embedding_1d(hidden_dim, arange(T))` 作为 1D PE，与 timestep embedding 并列相加。

```python
pos = torch.arange(T, device=x.device, dtype=torch.float32)
pos_emb = sinusoidal_embedding_1d(h.shape[-1], pos).to(h.dtype)
h = h + pos_emb.unsqueeze(0) + t_emb.unsqueeze(1)
```

### 9.2 tactile 段在 DiT 注意力里的 RoPE

**问题**：原实现给 tactile 段的 `freqs` 用 `torch.zeros(..., dtype=complex)` = `0+0j`。而 `rope_apply` 对 Q、K 做 `x * freqs`，相当于把 tactile 的 Q/K 直接清零，attention 里 tactile token 几乎失能。同时也没有给 tactile 提供任何与 video 对齐的时间坐标。

**修复**：tactile 段的 freqs 改为
- 时间维：用 `linspace(0, f-1, T_tac).round().long()` 把 tactile 位置映射到 video latent f 轴的同一坐标系，取 `dit.freqs[0]` 对应行 → tactile token i 与 video frame `round(i·(f-1)/(T_tac-1))` 共享 RoPE 时间位置。
- h/w 维：用 `1+0j`（单位复数，无旋转），保持 Q/K 数值不变。

这样 cross-attn 才能学到"tactile[i] 应该看 video frame i 附近的 token"。

### 9.3 首帧 anchor（"image + force_0 → video + force"）

**动机**：图像可观测但 force_0 不能从一张静止图反推（无法分辨"轻贴 2N"vs"夹紧 10N"）。把 force_0 作为已知 boundary condition 注入，可以把 12 维初始力的不确定性整段消除，且利用力序列高度自相关的特性收紧后验。和 video 那边"VAE 第 0 帧 image-cond clean"形成对称的 (image, force_0) 联合条件。

**实现**：跨三处统一开关 `tactile_anchor_first`。

| 位置 | 行为 |
|------|------|
| `model_fn_wan_video_joint_denoise` | 当 `tactile_anchor_first=True` 时，`tac_ts_flat = [0, t, t, ..., t]`，第 0 个 tactile token 的 AdaLN modulation 用 timestep=0；否则全部用当前 `t`（无 anchor 模式）。 |
| `FlowMatchJointDenoiseLoss`（训练） | `noisy_tactile[:, 0:1] = tactile_input[:, 0:1]` 保持 clean；loss 计算时 `noise_pred_tactile[:, 1:]` / `tactile_target[:, 1:]` 排除第 0 位（与视频侧 `[:, :, 1:]` 对称）；`inputs["tactile_anchor_first"] = True`。 |
| `WanTI2V5BJointDenoisePipeline.__call__`（推理） | `tactile_init` 接受 `(D,)` / `(1,D)` / `(1,1,D)`（仅 force_0）或 `(N,D)` / `(1,N,D)`（完整 GT）；前者 1..N-1 用噪声 + 第 0 帧 clean，后者整段加噪 + 第 0 帧覆盖回 clean。每个 `scheduler.step` 之后再次执行 `tactile[:, 0:1] = tactile_anchor`，并把 `tactile_anchor_first=True` 同时传给 posi/nega 分支。 |

**train/test 对齐说明**：训练侧只要 `tactile_input` 存在就一定 anchor，模型从未见过"有 tactile 但无 anchor"的输入分布。推理时若不传 `tactile_init`，model_fn 收到 `tactile_anchor_first=False`，会落到训练分布外，预测可能很差——这与 video 侧 image-cond 永远开启是一致的设计假设。如果以后需要支持"无 force_0 推理"，需要在训练时随机 dropout anchor（如 10% 概率把第 0 帧也加噪、`tactile_anchor_first=False`），让模型同时见过两种模式；当前实现按"force_0 永远已知"的部署假设来做。

---

## 10. Failure-frame 推理流水线（基于 `failure_frame_annotations.csv`）

把训练好的 joint denoise 模型回放到真实失败场景，验证「给定 fail 帧的图像 + 同一帧的力」能否预测出更合理的 rollout。流程分两步：

1. `scripts/generate_failure_clips.py` 从 fail 标注 CSV + 原始 hdf5 抽出推理输入；
2. `examples/CustomDataset/infer_tactile_stride5_192x256_joint_denoise_ti2v_5b.yaml` 配 `WanInferRunner` 跑 joint denoise，每个 fail 视频出 N 个 rollout（视频 + tactile）。

### 10.1 输入数据：从 fail 帧切 37 帧

CSV 来源（示例）：`/path/to/data/infer_logs/wipe_whiteboard_force_framestep5/failure_frame_annotations.csv`，列：

```
video, status, start_frame_0based
23.mp4, fail, 28
...
```

源 hdf5 在 `training_data/new_data/infer_logs/wipe_whiteboard_force_framestep5/<stem>.hdf5`，包含：

- `observations/images/cam_front` / `cam_high`：每帧 JPEG bytes
- `tactile/force_left` `(T,6)` + `tactile/force_right` `(T,6)` → 拼成 `(T,12)`，与训练 `tactile_dim=12` 对齐

脚本对每条 `status==fail` 的行：

- `start = start_frame_0based`，截 `[start, start+37)` 帧
- `cam_front` JPEG 解码 → resize 到 `192x256` → `<demo>/video.mp4`（6 fps）
- 力序列同区间 → `(37,12) float32` → `<demo>/force.npy`
- 在 `metadata_<task>.csv` 里复制 `--num_samples N` 行（共享同一份 video/force，仅 demo_id 后缀不同 `_n0`/`_n1`/…）；`WanInferRunner` 用 `seed_base + row_id` 给每行不同 seed，从而得到 N 个 rollout

> 只要 `start < T_source` 就保留：当 `start + num_frames > T_source` 时，**尾部用最后一帧/最后一行力重复 pad** 到 `num_frames`，日志里会标 `(+K pad)`。这意味着 CSV 里所有 fail 视频都会进 metadata，全部 32 条 fail 都能跑。  
> Pad 引入的尾段是「静止」假象，与训练分布有偏；如果只想看头几帧的 rollout 质量，pad 区段可以在事后忽略（`tactile_gt[K:]` 是常数，看 video 也能识别静止段）。

运行：

```bash
python scripts/generate_failure_clips.py \
  --csv /path/to/data/infer_logs/wipe_whiteboard_force_framestep5/failure_frame_annotations.csv \
  --src_dir /path/to/TACO/training_data/new_data/infer_logs/wipe_whiteboard_force_framestep5 \
  --out_dir /path/to/worldrl/wan_worldrl_stride5_192x256 \
  --task_name wipe_whiteboard_failure \
  --prompt "wipe the whiteboard" \
  --num_samples 10
```

输出：

```
<out_dir>/<task_name>/<stem>_s<start>/video.mp4   # 192x256, 6fps, 37 frames（首帧 = fail 帧）
<out_dir>/<task_name>/<stem>_s<start>/force.npy   # (37,12) float32（首行 = fail 帧的力）
<out_dir>/metadata_<task_name>.csv                # 与训练 metadata 同 8 列 schema，N 行/视频
```

主要参数：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--num_frames` | 每个 clip 的帧数 | 37（与训练 `num_frames` 一致） |
| `--num_samples N` | 每个 fail 视频在 metadata 里复制的行数 → 跑 N 次 inference 出 N 个 rollout | 1 |
| `--camera` | hdf5 里取哪一路相机 | `cam_front` |
| `--height / --width / --fps` | 输出视频尺寸/帧率 | 192 / 256 / 6 |
| `--prompt` | metadata 里写入的文本提示 | `wipe the whiteboard` |
| `--task_name` | 输出子目录名 + `metadata_<task>.csv` 文件名 | `wipe_whiteboard_failure` |

### 10.2 数据集如何把 fail 帧的力变成 anchor

`TactileVideoInferenceDataset`（`visuo_tactile_world_model/world_model/dataset/video_infer.py`）每个 item：

- `input_image` = `video.mp4` 第 0 帧 → **fail 帧的图像**
- 读 `force.npy`，按 `stats_path` 做 z-score（必须与训练 `tactile_stats_path` 同一份 `.npz`）
- `tactile_init = force_full[:1]` → **fail 帧的 12 维力**（归一化空间），作为 joint denoise 的 clean anchor
- `tactile_gt = force_full` 仅供事后比对/画图

### 10.3 Runner patch：把 tactile_init 透传给 pipeline

`visuo_tactile_world_model/world_model/runner/wan/wan_infer.py:WanInferRunner.run` 已扩展（与 `periodic_validation` 同步）：

- `if "tactile_init" in item: call_kwargs["tactile_init"] = item["tactile_init"]`
- `result = model(...)`；若返回 `(video, tactile_pred)`（`return_tactile: true`），把 `tactile_pred` 存为 `<video_basename>_tactile.npy`（位于归一化空间，需要物理量请按 `force_stats.npz` 反归一化）

### 10.4 推理 yaml 与启动

`examples/CustomDataset/infer_tactile_stride5_192x256_joint_denoise_ti2v_5b.yaml` 与训练 yaml 镜像：

- `model.class_path` = `WanTI2V5BJointDenoisePipeline`，`tactile_dim/hidden_dim/num_layers` 与训练完全一致
- `dataset` = `TactileVideoInferenceDataset`，`base_path` 指向脚本 `--out_dir`，`metadata_path` 指向脚本生成的 `metadata_<task>.csv`，`stats_path` 与训练同一份 `force_stats.npz`
- `runner.params.infer_kwargs`：`height: 192, width: 256, num_frames: 37, fps: 6, return_tactile: true`，`input_image_resize_mode: letterbox`

> ⚠️ `model.params.model_paths` 默认占位为基座 checkpoint。**实际使用前**改成训练产物的合并权重（含 `dit + tactile_tokenizer + tactile_head`，三者都在 `trainable_models` 里）。如果你按 §8 的方式把 `tactile_tokenizer.*` / `tactile_head.*` 单独 load，则 `model_paths` 仍可保留基座路径，由代码二次注入。

启动有两种方式：

**单卡：**

```bash
python run.py --config examples/CustomDataset/infer_tactile_stride5_192x256_joint_denoise_ti2v_5b.yaml
```

**多卡（推荐，每卡处理 `dataset[rank::world_size]`）：**

`WanInferRunner` 会读 `RANK` / `WORLD_SIZE` 环境变量并按步长切片 dataset；输出文件名以 `row_id` 起头（即 metadata 行号），不同 rank 永不冲突。`LIGHTEWM_RUN_ID` 让所有 rank 共享同一个 `logs/<config>/<run_id>/` 目录。

封装好的 launcher：

```bash
bash scripts/run_failure_infer_multigpu.sh --gpus 0,1,2,3
# 自定义 ckpt / 配置 / run_id
bash scripts/run_failure_infer_multigpu.sh \
  --gpus 0,1,2,3 \
  --config examples/CustomDataset/infer_tactile_stride5_192x256_joint_denoise_ti2v_5b.yaml \
  --ckpt logs/train/.../step-XXXX.safetensors \
  --run-id run_failure_n10_4gpu
```

每个 rank 一份 stdout/stderr 落到 `logs/<config>/<run_id>/launcher_logs/rank{0..3}.log`。

> 与 N=10 的关系：`generate_failure_clips.py --num_samples 10` 让 metadata 里每个 fail 视频复制 10 行；4 卡同时跑时 rank 0/1/2/3 各拿 1/4 行，总样本数不变，吞吐 ≈ 4×。

输出（默认 `output_dir = ./outputs/wipe_whiteboard_failure_joint_denoise_ti2v_5b` 或 launcher 设置的 `logs/<config>/<run_id>/`）：

```
000000__23_s28_n0__cam_front.mp4
000000__23_s28_n0__cam_front_tactile.npy   # (1, 37, 12) float32, 归一化空间
000001__23_s28_n1__cam_front.mp4
...
```

### 10.5 对齐校验清单

- [ ] 训练时是否传了 `tactile_stats_path`？是 → 推理 yaml 的 `stats_path` 必须指同一个 `.npz`，事后比较物理量需反归一化
- [ ] 推理 `num_frames / height / width / tactile_dim / tactile_hidden_dim / tactile_num_layers` 与训练完全一致（不一致会导致 `tactile_tokenizer` shape mismatch）
- [ ] `model_paths` 已指向含 tactile 模块的训练 ckpt，或代码侧已显式 `load_state_dict` 进 `pipe.tactile_tokenizer / pipe.tactile_head`
- [ ] CSV 里 fail 行的 `start_frame_0based + num_frames <= T_source`，否则该行被 skip

---

## 11. 常见问题

**Q：`loss/tactile` 不下降？**
1. 确认 cache `.pth` 中确实有 `tactile_input`（用任意 pth 打开 `inputs_shared.keys()` 检查）；
2. 确认 `trainable_models` 包含 `tactile_tokenizer,tactile_head`；
3. 确认 `model.class_path` 是 joint 版本，否则 `tactile_tokenizer is None` 时 model_fn 走视频-only 分支；
4. tx/tz 量级到 ±80 N·m 时考虑加归一化（参考 §6 与原 README）。

**Q：`tactile_dim` 怎么改？**
改 YAML 里 `tactile_dim`（同时确保 `force_sequence.npy` 的最后一维匹配）。Pipeline 在 `from_pretrained` 时按这个值实例化 tokenizer / head。

**Q：单卡 debug？**
打开 `.vscode/launch.json` → 运行 `Debug: train_full worldrl (single GPU, visuo_tactile_world_model.world_model_backup)`。该配置已固定使用 `visuo_tactile_world_model.world_model_backup` 的 Python 解释器，绕过 `accelerate launch` 直接 `debugpy` 启动 `run.py`，能命中 `WanTrainingModule.forward` / `FlowMatchJointDenoiseLoss` 内部断点。

---

## 11. 批量推理（WanInferRunner）

`WanInferRunner` 支持对整个数据集批量推理并保存视频（视频生成只走视频分支，不产出 tactile）。如需同时推理出触觉序列，需要改用手工调用 `pipe(return_tactile=True, ...)` 的方式（见 §8）。

**Infer yaml 示例**：

```yaml
task: infer

runner:
  class_path: visuo_tactile_world_model.world_model.runner.wan.wan_infer.WanInferRunner
  params:
    output_dir: ./outputs/joint_denoise_infer
    fps: 6
    quality: 5
    seed: 0
    # "tail" 精确命中每个视频的 max(0, N - window_size) 帧
    # 对应训练中 context_window_tail_align 生成的 tail window 起点
    conditioning_frame_fractions: [0.0, 0.25, 0.5, 0.75, "tail"]
    infer_kwargs:
      negative_prompt: ""
      height: 192
      width: 256
      num_frames: 37
      cfg_scale: 1.0
      num_inference_steps: 30
      tiled: true

model:
  class_path: visuo_tactile_world_model.world_model.model.wan.pipeline_ti2v_5b_joint_denoise.WanTI2V5BJointDenoisePipeline
  params:
    device: cuda
    torch_dtype: bfloat16
    tokenizer_path: /path/to/google/umt5-xxl
    model_paths:
      - /path/to/checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors
      - /path/to/checkpoints/wan-t2v-5b/models_t5_umt5-xxl-enc-bf16.pth
      - /path/to/checkpoints/wan-t2v-5b/Wan2.2_VAE.pth

dataset:
  class_path: visuo_tactile_world_model.world_model.dataset.video_infer.VideoInferenceDataset
  params:
    base_path: /path/to/worldrl/wan_worldrl_stride5_192x256
    metadata_path: /path/to/worldrl/wan_worldrl_stride5_192x256/metadata_taco_joint.csv
    video_key: video
    prompt_key: prompt
```

**输出文件**：

| 文件 | 说明 |
|------|------|
| `<row_id>__<demo_id>__<camera>__f<frame>_p<pct>.mp4` | 生成视频（浮点 fraction） |
| `<...>__f<frame>_tail.mp4` | 生成视频（`"tail"` fraction，对应视频末尾 window） |
| `<...>_compare.mp4` | 左半 GT、右半生成的并排对比视频（仅当 `video_path` 可读时生成） |

```bash
python run.py --config <your_joint_infer_yaml>
```
