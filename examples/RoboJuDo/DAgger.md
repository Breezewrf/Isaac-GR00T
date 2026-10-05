# DAgger 实现对比：GR00T、RLInf 与 LeRobot

本文对照当前 GR00T 离线 DAgger、RLInf 真机在线 HG-DAgger，以及 LeRobot 的 HIL/DAgger 采集与微调示例。表中的超参数来自具体示例，不代表 DAgger 的通用要求。当前 GR00T 的运行方法见 [README.md](README.md)。

## 1. 采集数据与训练样本

当前 GR00T 支持用 [dagger_data.yaml](dagger_data.yaml) 在训练时混合原始 SFT 和完整 DAgger 数据集：SFT 不要求接管字段，DAgger 使用现有标签筛选专家窗口。两个数据集保留各自的 episode、task 和视频索引，复用 SFT checkpoint 归一化统计；无需物理合并或写入合并 `stats.json`。比例针对筛选后的训练样本，示例值不是普通训练默认值。使用方法见 [README](README.md#mix-original-sft-demonstrations-and-dagger-interventions)。

| 项目 | LeRobot 默认 DAgger | LeRobot `record_autonomous=True` DAgger | 当前 GR00T 离线 DAgger | RLInf 在线 HG-DAgger |
| --- | --- | --- | --- | --- |
| 保存的帧 | 只保存人工修正；策略自主运行时不记录 | 保存策略和人工帧，分别标记 `intervention=False/True` | 保存完整成功 rollout，包含策略和专家帧，逐帧标记 `expert_applied`、`action_source` | 保存完整成功 rollout，包含策略和专家帧，逐帧标记 `intervene_flag` |
| Episode 边界 | 每段完成的人工修正单独成为一个 episode | 连续记录；按视频大小估算的时长切分，人工修正期间推迟切分 | 完整成功 rollout | 完整成功 rollout |
| 失败案例 | DAgger 采集策略没有自动“只保留成功”开关 | DAgger 采集策略没有自动“只保留成功”开关 | 本项目在采集时丢弃失败案例 | 示例通过 `only_success=True` 丢弃失败案例 |
| 训练样本原则 | 数据集只含人工修正帧；普通训练器从这些帧取样，以人工动作为监督目标 | 数据集同时含策略帧和人工帧；普通训练器从两者取样，`intervention` 标签不会自动筛除策略动作 | 原始示教正常取样；DAgger 只训练专家动作窗口，按 YAML 比例混合 | DAgger 只训练专家动作窗口 |
| 动作窗口末尾 | 使用所选模型的普通 chunk 处理 | 使用所选模型的普通 chunk 处理 | 16 步都必须是真实专家动作；不足 16 步的窗口不进入训练 | 10 步窗口允许 episode 末尾 padding；所有**非 padding**动作都必须是人工接管动作 |
| 数据更新 | 采集结束后使用固定数据集微调 | 采集结束后使用固定数据集微调 | 采集结束后使用固定数据集微调 | 采集期间更新；只采样最近 50,000 个符合条件的窗口起点 |

**当前 GR00T 与 RLInf 在采集内容和专家监督原则上基本一致。** 两者都保留完整成功 rollout，策略帧可作为观测上下文，但策略动作不会成为专家监督目标。主要差别是离线固定数据集与在线滚动数据集，以及动作窗口末尾是否允许 padding。16 步与 10 步是各自示例的模型配置，并非 DAgger 定义不同。当前 GR00T 的两个 episode 末尾都至少有 15 帧策略动作，因此即使允许 RLInf 式 padding，也不会增加专家有效窗口；有效起点仍为 868。[GR00T 采样器](../../gr00t/data/dataset/sharded_single_step_dataset.py)、[RLInf 在线采样器](https://github.com/RLinf/RLinf/blob/main/rlinf/data/datasets/dagger/dataset.py)、[RLInf 示例配置](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/realworld_pnp_dagger_openpi.yaml)

LeRobot 当前代码的 `record_autonomous` 默认值为 `False`：每段人工修正保存为一个 episode。普通训练器无需识别“修正”标签；它读取这些 episode 中的帧，而这些帧在采集时就已限定为人工修正。设为 `True` 后才会在同一段连续记录中保存自主帧与人工帧；暂停期间不记录。其 [HIL 指南](https://huggingface.co/docs/lerobot/hil_data_collection)对“同时记录两类帧”的概述，对应可选的连续记录模式，而非代码默认值。另外，**原始 SFT 示教数据不会自动加入**。两种模式都需要先把它与采集数据合并，训练命令再指向合并后的数据集。[DAgger 采集配置](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/rollout/configs.py#L207-L263)、[只记录修正的实现](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/rollout/strategies/dagger.py#L518-L675)、[连续记录的实现](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/rollout/strategies/dagger.py#L340-L512)

## 2. RLInf：普通真机 SFT 与在线 HG-DAgger

以下比较 RLInf 的 Franka bin-relocation SFT 示例与真机 PnP HG-DAgger 示例。[SFT 配置](https://github.com/RLinf/RLinf/blob/main/examples/sft/config/realworld_bin_relocation_sft_openpi.yaml)、[HG-DAgger 配置](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/realworld_pnp_dagger_openpi.yaml)

| 配置 | 普通 SFT | 在线 HG-DAgger |
| --- | --- | --- |
| 入口 | `run_vla_sft.sh realworld_bin_relocation_sft_openpi` | `run_realworld_async.sh realworld_pnp_dagger_openpi` |
| 训练器与数据 | `SFTRunner` 读取 `data.train_data_paths` 指定的固定 LeRobot 数据集 | `AsyncEmbodiedRunner` 同时运行环境与 rollout；actor 从滚动 LeRobot 数据集中取样 |
| 模型任务 | `openpi.task: sft` → `Pi0` | `openpi.task: dagger` → `Pi0DAgger` |
| 监督动作 | 原始示教动作 | 策略到访状态上的人工接管动作 |
| 样本筛选 | 普通 SFT 数据加载 | `only_success=True` 保留成功 episode；`only_save_expert=True` 只暴露专家有效的 chunk 起点 |
| 学习率 | `2.5e-5` | `1e-5` |
| Adam β₁ / β₂ / ε；weight decay | `0.9` / `0.95` / `1e-8`；`1e-10` | 相同 |
| warmup | cosine；`1000` 步 | cosine；`20` 步 |
| 调度器 `total_training_steps` | `30000` | `1000` |
| 梯度裁剪 | `1.0` | `4.0` |
| micro / global batch size | `4` / `32` | `32` / `32` |
| 运行上限 | `max_steps: 2000` | `max_epochs: 8000`、`max_steps: -1` |
| 定期保存间隔 | `save_interval: 100` | `save_interval: -1` |
| FSDP | `no_shard`、`use_orig_params: True` | `no_shard`、继承模板的 `use_orig_params: False` |
| 推理积分步数 | SFT Pi0 模板 `num_steps: 5` | Embodiment Pi0 模板 `num_steps: 4` |

`total_training_steps` 是学习率调度器的长度，并不单独决定实际运行步数。两种 micro batch 配置的 global batch 都是 32；`num_steps` 是推理设置，不改变示例的 10 步执行 chunk。[SFT Pi0 模板](https://github.com/RLinf/RLinf/blob/main/examples/sft/config/model/pi0.yaml)、[Embodiment Pi0 模板](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/model/pi0.yaml)

两套示例都使用 OpenPI Pi0、`pi0_realworld`、7 维动作、两个图像输入、10 步执行 chunk；默认不使用 LoRA，也没有 value head。模型变体同为 Gemma 2B / Gemma 300M。`Pi0DAgger` 的前向训练仍调用 Pi0 的 SFT 监督损失；DAgger actor 对 OpenPI 启用 action chunk loss，并跳过 advantage 计算。因此 DAgger 的关键区别在采集、标签、样本筛选和更新时机，而不是引入 PPO 式损失。[模型构造](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/__init__.py)、[Pi0DAgger](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/tasks/dagger.py)、[DAgger actor](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/actor/fsdp_dagger_policy_worker.py)

HG-DAgger 配置中的 `openpi.train_expert_only: False` **不是**专家接管样本筛选开关；真正控制筛选的是 `algorithm.dagger.only_save_expert`。模型构造代码只在 `task: rl` 分支读取前者来决定是否冻结 VLM，`task: dagger` 不因该字段冻结 VLM。配置里虽有 `gamma`、`adv_type`、`value_lr`，但此示例没有 value head，DAgger actor 也跳过 advantage 计算；这些字段不能说明它在进行 PPO 训练。[模型构造与冻结逻辑](https://github.com/RLinf/RLinf/blob/main/rlinf/models/embodiment/openpi/__init__.py)、[DAgger actor](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/actor/fsdp_dagger_policy_worker.py)

### RLInf 是否有“采集后离线训练 HG-DAgger”的现成示例？

在所查 RLInf 源码中没有找到独立的示例或训练入口。[真机指南](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/hg-dagger.html)中的单独 SFT 步骤用于**在线 HG-DAgger 之前**初始化 student。在线 DAgger actor 可以在恢复运行时重载已归档的 LeRobot shard，但这仍属于会启动环境和 rollout worker 的在线训练器。普通 SFT worker 不使用 `RollingLeRobotDataset` 的 `require_all_intervene` 筛选；直接把混有策略动作的归档交给普通 SFT，不能自动复现在线 HG-DAgger 的专家窗口训练。[归档恢复与采样代码](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/actor/fsdp_dagger_policy_worker.py)、[普通 SFT worker](https://github.com/RLinf/RLinf/blob/main/rlinf/workers/sft/fsdp_vla_sft_worker.py)、[在线 runner](https://github.com/RLinf/RLinf/blob/main/rlinf/runners/async_embodied_runner.py)

## 3. LeRobot：普通 Pi0 训练与 HIL 微调

以下依据 LeRobot 的 [`e0d5021` 版本](https://github.com/huggingface/lerobot/tree/e0d50211ef236143ae867228662b7dfaba554f02)。`--strategy.type=dagger` 选择的是 **rollout/采集策略**，没有切换到独立的 DAgger 训练器。HIL 指南仍调用普通 `lerobot_train.py`，从已有策略权重继续微调。[HIL 指南](https://huggingface.co/docs/lerobot/hil_data_collection)

| 配置 | 指南中的普通 Pi0 示教训练 | 指南中的 HIL 微调 |
| --- | --- | --- |
| 训练入口 | `lerobot_train.py` | 相同 |
| 模型权重 | 初始化 Pi0 | `--policy.pretrained_path` 加载已训练策略 |
| 数据 | 原始示教数据集 | 指南要求使用原始示教与 HIL 数据的合并数据集 |
| 命令指定的训练步数 | `50000` | `20000` |
| 命令指定的 batch size | `32` | 未指定，因此使用训练配置默认值 `8` |
| DAgger 专用学习率、损失或专家筛选开关 | 无 | 指南命令中也没有 |

- 在未修改 Pi0 配置的前提下，两次训练都使用相同的 [Pi0 优化器预设](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/policies/pi0/configuration_pi0.py#L88-L168), 学习率并没有因为叫 DAgger 微调就自动降低。在使用相同 Pi0 配置且没有额外覆盖参数时，两次训练沿用 Pi0 的优化器预设；总步数变为 20000 后，学习率调度的时长会相应缩放。实际值仍应以加载的模型配置和最终训练配置为准。默认采集模式与`record_autonomous=True` 之间，训练命令没有区别。

- 普通训练配置默认启用策略的优化器预设，`sample_weighting=None`；现有样本加权选项是 `rabc` 和 `uniform`，没有按 `intervention` 标签筛选的选项。[训练配置](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/configs/train.py#L108-L179)、[样本加权实现](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/utils/sample_weighting.py)

- 指南要求的“合并数据集”必须预先准备：当前 [训练配置](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/configs/train.py#L309-L316)不接受多个 dataset ID 列表，而 [dataset tools](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/datasets/dataset_tools.py#L278-L321)提供 `merge_datasets`。指南的训练命令只指向一个 HIL dataset ID；只有该 ID 已经指向合并结果时，命令才真正使用“原示教 + HIL 数据”。

[普通训练采样器](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/scripts/lerobot_train.py#L324-L345)按 episode 边界选择并打乱帧，不检查 `intervention`。Pi0 使用[未来动作索引](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/policies/pi0/configuration_pi0.py#L170-L180)构造动作窗口；[数据读取器](https://github.com/huggingface/lerobot/blob/e0d50211ef236143ae867228662b7dfaba554f02/src/lerobot/datasets/dataset_reader.py#L305-L324)在 episode 边界截取索引并返回 `<action_key>_is_pad`。padding 本身不会筛掉连续记录中的策略动作，也不会保证整个窗口都由人工接管。若要复现当前 GR00T 的离线专家窗口训练，需要额外预处理数据或实现专门的采样器。
