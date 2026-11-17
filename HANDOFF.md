# FM-Receiver 项目交接

交接于 2026-11-16。组会已汇报，老师认可方案方向。下一项工作是在 Linux 服务器实现并运行传统 PUSCH 软件基线。这里记录截至交接时的配置和计划。原始历史讨论保留在本地，原文中的部分文件路径来自 Mac。

## 研究目标与当前进度

课题从 FM-Receiver 出发，在固定的小规模链路上验证传统接收机、Transformer 外接收机和 G-ECCT，再评估软件仿真到 X410 有线实测的适应效果。传统 B0、TDL-A、两个种子的 B1/G-ECCT 独立训练、联合微调与统一 GPU 评估均已完成。本轮 B1 在有效比较点优于 B0，当前 B2 没有性能增益；结果见 [正式报告](reports/2026-11-16/README.md)。X410 的离线接口已验证，真实 RF 采集尚未进行。运行方式见 [README.md](README.md)，完整记录见 [VALIDATION.md](VALIDATION.md)。

| 首轮项目 | 约定 |
| --- | --- |
| 链路 | 单用户、单流、1Tx/1Rx，4 PRB（48 个分配子载波），14 个 OFDM 符号，15 kHz 子载波间隔 |
| 编码与调制 | NR LDPC/TB，QPSK，MCS 表 1、索引 9 |
| 软件信道 | 先无噪声直通，再扫 AWGN，首轮验证通过后加固定 TDL-A |
| 传统接收 | LS 估计及线性插值 → LMMSE 均衡、软解调 → 常规 TB 译码 |
| 后续外接收网络 | FFT 后的接收栅格 (Y) 与 LS 信道估计 → 2 层、128 维、8 头 Transformer → LLR → 同一常规 TB 译码器 |

TB 长度、编码长度、DMRS 位置和每个数据 RE 的排列以生成器输出为准。日后用 X410 时，采样率、FFT 点数、CP 样本数和端口随硬件接口验证后记录。

当前 Sionna 2.2 回读结果：每 TB 848 信息比特、1248 Mapper 比特、624 数据 RE，DMRS 符号索引为 `[2]`。软件网格使用 48 点 FFT、4 个 CP 样本、720 ksample/s；这些不是已冻结的 X410 参数。代码按 NRX 的配置、链路、接收机和评估分工组织，使用独立的 PyTorch／Sionna 2.2 环境。

## 当前环境与协作方式

使用本项目独立 Conda 环境 `./.conda/fm-receiver`，通过 `bash scripts/setup_env.sh`
创建，随后执行 `conda activate ./.conda/fm-receiver`。Python 和标准库均安装在该目录，
不复用个人 `deepreceiver` 等环境的解释器。此前 `.venv` 只保留为历史验证环境。

Git 提交 `environment.yml`、`conda-linux-64.lock`、`requirements.lock.txt`、安装／导出脚本、
源码、配置、测试与文档；环境目录、缓存、结果、数据和权重由 `.gitignore` 排除。
新增实验沿用现有模块分工，依赖变更应更新锁文件并在新环境中验证。

独立 Conda 环境已完成全套 CPU 基线验证，结果见 `results/b0_conda_final/`；
干净源码副本的独立安装、测试及路径检查见 `results/conda_portability/verification.json`。
独立 Conda GPU 结果为 `results/20261116T091432103629Z_evaluate/` 与
`results/20261116T104302933134Z_evaluate/`，两次均通过完整默认配置验收。
运行记录确认解释器、`prefix` 和 `base_prefix` 均指向 `.conda/fm-receiver`，
实际设备为 RTX 3090，直通、AWGN、同观测 CPU／GPU 对照及 64 帧样本重放均通过。
两次源码、配置、AWGN 输入哈希和全部错误计数一致，核对记录见
`results/b0_conda_gpu_acceptance.json`。首轮 B0 已完成，无需继续补跑原有 AWGN 验收。
详见 [VALIDATION.md](VALIDATION.md)。

## 首轮任务（已完成）

1. 检查服务器上的 Python、PyTorch、Sionna 和可用设备。在项目独立环境里记录实际依赖版本。用 Sionna 官方 PUSCH 的 Hello World 确认环境，再按上表固定通信配置。首个小批次可以在 CPU 上做。
2. 在一个可重复运行的 Python 脚本里用 `PUSCHTransmitter` 生成波形和信息比特，用默认 `PUSCHReceiver` 恢复比特并返回 TB CRC 状态。无噪声直通时逐位比较信息比特，验收为本批次零错误、CRC 全通过。如果接收机数值计算要求正的噪声方差，可给接收机一个小的正数，发射栅格仍直通。
3. 用同一配置扫几个 AWGN 信噪比点，记录随机种子、错误比特数、总比特数、错误块数、总块数及 CRC 失败数，导出 BER 和 BLER。检查曲线整体趋势，并保存运行命令和最终生成的配置。
4. 基线通过后，从这条链路导出一小批带帧标识的样本。后续外接收机输入 (Y)、LS 估计，训练标签取各数据位置实际送入 QPSK 映射的比特。保存数据位置掩码和排列方式，并保留原始 TB 信息比特用于译码后的 BER/BLER 比较。

实施参考：[Sionna 安装](https://nvlabs.github.io/sionna/installation.html)、[PUSCH 官方教程](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html#a-hello-world-example)、[PUSCHReceiver](https://nvlabs.github.io/sionna/phy/api/nr/sionna.phy.nr.PUSCHReceiver.html)。

## 后续顺序

原定软件实施与 GPU 流水线已完成。运行目录 `results/pipelines/20261116T134137/` 的 14 个任务
全部完成，两个种子的正式评估共 108 行结果，来源、同观测输入及误码计数均核验通过。
完整测试 22 项全部通过，GPU 项没有跳过。用户已开放 full access，工具可直接操作 CUDA 和
602-2 SSH，目前不需要用户输入终端指令。

G-ECCT 的 64 帧门限由原检查点续训后在 1250/1150 步通过，未放宽门限；正式 B2 仍弱于 B1。
这属于本轮研究结果，不应覆盖失败记录或通过改测试数据宣称提升。补充诊断和后续消融方向见
[接续记录](IMPLEMENTATION_STATUS.md)及运行目录的 `diagnostics/NOTES.md`。

本轮冻结源码归档、环境锁文件和结果均保留在数据盘。新增实验使用独立配置和输出目录，
维持配置、模型、训练、接收机、评估和硬件适配的模块边界。当前没有待恢复的训练任务。
X410 后续按照 [接口清单](docs/X410_INTERFACE.md)核对设备、同步、采样率和事件，
先传统接收核验真实采集的帧与标签对齐，再考虑训练和适应。

## 可以发给服务器 Codex 的第一条消息

> 请读 HANDOFF.md。检查本目录和 Linux 上的 Python/PyTorch/Sionna 环境，按首轮配置实现并实际运行传统 PUSCH 基线。先验证无噪声直通，再跑 AWGN，保存配置、运行命令、BER/BLER 和错误计数。这一轮先交付软件基线。
