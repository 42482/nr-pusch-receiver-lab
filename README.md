# NR PUSCH Receiver Lab

本项目先实现 B0：**NR PUSCH 发射 → 直通／AWGN → LS 与线性插值 → LMMSE 与软解调 → TB 译码**。
工程分工参考 [NVlabs/neural_rx](https://github.com/NVlabs/neural_rx)；实现采用 PyTorch 和 Sionna 2.2。
配置依据见 [HANDOFF.md](HANDOFF.md)；原始讨论记录保留在本地。
实现与实测状态见 [VALIDATION.md](VALIDATION.md)。
当前 B0 的直通／AWGN 已在项目独立 Conda 环境完成 CPU 和 RTX 3090 GPU 验收，
两次 GPU 运行的输入哈希及误码计数一致。TDL-A GPU、两个种子的 B1/G-ECCT 训练、
联合微调及正式同观测评估现已完成；本轮 B1 在有效比较点改善，当前 B2 没有性能增益。
[正式结果与曲线](reports/2026-11-16/README.md)保留完整计数，
[接续记录](IMPLEMENTATION_STATUS.md)说明下一轮研究与源码版本约定。
后台运行与接续见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)，
实测接入契约见 [X410_INTERFACE.md](docs/X410_INTERFACE.md)。
同伴拉取、版本选择和首次上传见 [协作说明](docs/COLLABORATION.md)。
当前 `v0.1.1` 已通过 24 项测试；历史实验对应 `frozen-20261116` 标签。

## 克隆、独立 Conda 环境与运行

当前锁定配置支持 **Linux x86_64、Python 3.11.15、PyTorch 2.11.0+cu128、Sionna 2.2.0**。
先安装并初始化 Conda（本机管理器版本为 25.11.1）。GPU 运行还需要兼容 CUDA 12.8 的
NVIDIA 驱动；GPU 硬件和宿主机驱动由运行机器提供。CPU 检查使用同一软件环境。

```bash
git clone <仓库地址> nr-pusch-receiver-lab
cd nr-pusch-receiver-lab
bash scripts/setup_env.sh
conda activate ./.conda/fm-receiver
python -m unittest discover -s tests -t . -v
python scripts/evaluate.py --device cuda:0
```

已有源码时从 `bash scripts/setup_env.sh` 开始。环境目录是本仓库下的 `.conda/fm-receiver`，
Python、标准库和运行库均由锁定发行包安装。包缓存可以复用下载文件，运行时不依赖缓存、
其他项目环境或原作者的解释器。安装后检查 `sys.prefix == sys.base_prefix`、解释器位置、
库导入、CPU 复数运算和 `pip check`，本机检查记录保存在 `.cache/environment-check.json`。

| 文件 | 用途 |
|---|---|
| `environment.yml` | 可读的完整环境入口，含 Python 版本、Conda 来源及对 pip 锁文件和项目安装的引用 |
| `conda-linux-64.lock` | Conda 基础包的准确构建、官方 URL 和 SHA256，固定 Python 与平台库 |
| `requirements.lock.txt` | pip 科研依赖及构建工具的准确版本，指定 PyPI 和 PyTorch CUDA 12.8 来源 |
| `pyproject.toml` | 本项目的软件依赖与安装入口；CUDA 构建选择由对应锁文件固定 |

安装顺序固定为 Conda 基础环境、pip 依赖、项目 editable 安装。安装器使用 Conda 的复制模式，
不克隆已有环境；已有环境必须带本项目创建标记，否则拒绝复用。环境目录和缓存均位于项目盘。
可用 `--prefix PATH` 选择另一个全新目录。安装器不改写 Git 中的定义和锁文件。

需要改依赖时，修改 `environment.yml`／`pyproject.toml` 并在新的临时环境中验证，再显式导出：

```bash
bash scripts/export_env.sh /path/to/validated-conda-prefix
# 审查两个锁文件的差异，按新锁文件重建环境并测试后提交 Git。
```

锁文件变化后使用新的 `--prefix` 重建，避免在旧的混合 Conda／pip 环境中继续叠加安装。
离线安装支持 `--offline --wheelhouse PATH`；需备齐 Conda 缓存及 pip wheel。
也可用 `--conda-lock FILE` 指向包含相同发行包及 SHA256 的本地镜像清单，本地路径清单留在
缓存目录中。Git 中的主锁文件保持公开下载 URL，不写入作者路径。

`--device cpu` 执行 CPU 检查，`--device auto` 允许自动选择。默认 `cuda:0` 不可用时直接报错，
运行记录保存实际设备。此前 `.venv` 仅对应历史验证记录，当前安装流程使用上述独立 Conda 环境。

## Git 中保存什么

提交源码、实验配置、测试、文档、上述环境定义和锁文件。`.gitignore` 忽略环境目录、下载缓存、
结果、权重、原始数据和本地配置；数据忽略规则只针对仓库根目录，不会漏掉 `src/fm_receiver/data/`
源码。B0 的样本和结果由固定种子重新生成。其他人克隆后运行安装脚本即可重建锁定的软件环境。

## 固定实验配置

唯一配置源为 `config/b0.toml`，采用有类型的 TOML 读取，未知字段与错误类型会被拒绝。
CLI 覆盖值保存在每次运行的 `requested_config.json`，库回读值保存在
`resolved_config.json`；评估时不会隐式覆盖为原 NRX 的 132 PRB。

| 项目 | 取值 |
|---|---|
| 用户、层、天线 | 1 用户、1 层、1Tx／1Rx |
| 资源 | 4 PRB、48 子载波、14 OFDM 符号、15 kHz、normal CP |
| 编码与调制 | NR TB／LDPC、QPSK、MCS 表 1／索引 9 |
| 目标码率 | 679/1024；实际信息码率另行记录 |
| DMRS | A、type 1、position 2、length 1、additional position 0、port 0、CDM groups 2 |
| 接收 | LS＋线性插值、LMMSE、max-log、20 次 BP／boxplus-phi |
| 精度 | float32／complex64 |

当前生成器回读：**848 信息比特／TB，1248 Mapper 比特／TB，624 数据 RE，DMRS 符号索引 `[2]`**。
实际信息码率为 848/1248；TB CRC、LDPC 母码及速率匹配参数见运行配置，不将上述长度写死在收发程序中。
软件网格为 48 点 FFT、4 个 CP 样本、720 ksample/s；这些值仅描述本轮软件对象，硬件参数仍待 X410 接口验证。

AWGN 横轴是**数据符号 Es/N0**。QPSK 数据符号平均能量为 1，复噪声方差为
`10**(-EsN0_dB/10)`，实部和虚部各占一半。图中零错误点以 `1/N` 作为显示位置，使用三角
标记注明“观察到零错误”；该位置是显示分辨率，不是误码率估计或置信上界，CSV 保存真实计数。

## 验证与复现

完整 `--scenario all` 默认依次执行：

1. 官方默认 PUSCH Hello World（其默认 MCS 为 14，仅作环境检查）。
2. 固定首轮配置，在不加噪的网格上累计验证 256 TB；接收机使用 `1e-6` 的数值噪声方差。
3. 与直接调用默认接收机逐位比较，并验证 Mapper 标签、OFDM 时域往返。
4. 扫描 `Es/N0=[-4,-2,0,2,4,6,8] dB`；每点至少 256 TB，达到 100 错误块或 4096 TB 停止。
5. 检查复噪声及实虚部功率，重建固定种子的小批次并比较输入和译码结果。
6. 用独立种子 43 在 2 dB 导出 64 帧样本，重新读取、重映射并重放接收机。

```bash
# 单独检查无噪声直通
python scripts/evaluate.py --device cpu --scenario passthrough

# 独立导出样本
python scripts/export_samples.py --device cuda:0

# 指定全新的结果目录，重复运行并检查输入哈希和错误计数
python scripts/evaluate.py --device cpu --output results/cpu_first
python scripts/evaluate.py --device cpu --output results/cpu_repeat --compare-run results/cpu_first
```

`--compare-run` 要求同一设备、相同生效配置和源文件版本，并逐项比较 AWGN 输入 SHA256、
比特／块错误计数与 CRC 失败数。跨 CPU／GPU 使用同一批观测检查直通比特、CRC 和信道估计，
不要求不同设备的随机发生器产生相同噪声。

支持 `--batch-size`、`--passthrough-tbs`、`--min-tbs`、`--max-tbs` 做小规模调试，
修改统计规模的运行会标记为非正式配置。输出目录必须是新目录，避免覆盖结果。
入口通过 editable 安装解析模块与配置位置，不依赖当前工作目录；也可使用安装生成的
`pusch-evaluate`、`pusch-export` 命令。

每次运行保存以下内容：

- `manifest.json`：命令、覆盖参数、执行设备、依赖检查、验收状态及错误原因。
- `config.toml`、`requested_config.json`、`resolved_config.json`：原始、生效及库生成配置。
- `environment.json`、`requirements.freeze.txt`、`source_hashes.json`：依赖与源码版本追溯。
- `environment/`：本次运行的 Conda 定义、双锁文件及项目依赖定义，均纳入来源哈希。
- `metrics.csv`／`metrics.json`：信噪比、实际噪声方差、比特错误／总比特、块错误／总块、CRC 失败数、BER／BLER。
- `ber_bler.png`、`checks.json`、`run.log`：曲线、正确性检查及运行日志。
- `samples/`：NPZ 样本、JSON 字段说明和保存后重放检查。

命令执行成功时 `status=completed`；完整默认配置、实际 CUDA 运行、全部检查及依赖检查通过时，
才会置 `milestone_accepted=true`。CPU 运行始终保留 GPU 待验收状态。失败以非零退出码返回，
已产生的计数和错误日志保留。

## 数据与后续扩展

| 分层 | 职责与后续接入点 |
|---|---|
| `config` / `builders` | 配置读取与运行组件创建分开；由配置选择通信与算法组件 |
| `e2e_model` / `channels` | 生成载荷、发射与软件信道；可只生成观测，交给多个接收机 |
| `sionna_adapter` | 集中所有 Sionna 内部属性访问和 Mapper 输入采集，升级版本时在此核对 |
| `receivers` | B0 封装和 B1/B2 神经接收适配器，共用观测与输出契约 |
| `decoders` / `models` | 共用 TB 译码器、外接收 Transformer、G-ECCT；NR 编码布局集中在适配层 |
| `data` / `hardware` | 观测与标签分离、样本重放、X410 IQ 文件与离线前端 |
| `training` / `orchestration` | 训练、验证与检查点；独立的多机/Screen 调度 |
| `evaluation` | 统一计数、停止条件、对照、结果及绘图 |

接收机只接收 `ObservationBatch(frame_ids, y, no, layout_id)`，返回
`ReceiverResult(tb_hat, tb_crc, llr, h_hat, err_var)`。发送真值在独立 `TxLabels` 中。
推理模块不接收 TB 真值、Mapper 标签或仿真真实信道。

导出的 `mapper_bits` 来自发射链路 Mapper 的真实输入，包含速率匹配、交织和扰码。
数据 RE 按零起始的“OFDM 符号优先、子载波次之”排列，每个 RE 对应两位 QPSK 标签。
`data_re_indices`、`data_mask`、`resource_type_grid` 和 `pilot_grid` 明确区分数据、导频与保留位置；
`pilot_grid` 包含实际导频值及保留位置上的零。

LLR 采用 `log(P(1)/P(0))`，正值倾向于 1。样本中的 LLR 是进入 TBDecoder 之前、与 Mapper
比特排列对齐的软信息。后续 G-ECCT 所需的解扰、解交织、速率恢复与校验矩阵列对应关系，
已在译码适配层验证，并由 oracle、校验子和门限记录追溯。

B1 增加 Transformer 模型和接收适配器，复用 B0 的 TB 译码组件；B2 再接入 G-ECCT。
接收机之间比较同一批观测，复用统一统计程序。X410 前处理最终也输出相同观测接口；噪声方差
应来自实测估计。训练循环位于独立 `training` 模块，不向评估脚本或通信组件堆入训练逻辑。
本轮 64 帧仅供标签与小样本过拟合检查；正式训练、验证、测试按不同载荷、种子或采集批次分开。

## 参考依据

- [Sionna PUSCH 官方教程](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html)
- [PUSCHReceiver](https://nvlabs.github.io/sionna/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html)
- [TBDecoder](https://nvlabs.github.io/sionna/phy/api/nr/sionna.phy.nr.TBDecoder.html)
- [Sionna Demapper 的 LLR 定义](https://nvlabs.github.io/sionna/phy/api/mapping/Demapper.html)
- [Conda 环境创建与复现](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

本项目参考 NRX 的职责划分，通信算子调用已安装 Sionna；原 NRX 仓库作为独立参考工程保留。
