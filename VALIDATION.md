# 首轮 B0 实施与验证记录

可随 Git 查看的是 [共享结果](reports/2026-11-16/README.md)。下文 `results/` 路径是服务器本地历史日志，不包含在共享仓库中。

记录日期：2026-11-16。**首轮 B0 的直通／AWGN 软件实现、独立 Conda 环境 CPU 与
RTX 3090 GPU 验证均已完成；两次完整 GPU 运行的输入哈希及误码计数完全一致。**

## 实现与环境

代码按 NRX 的配置、链路、接收机和评估分工实现，核心模块、运行命令及扩展接口见
[README.md](README.md)。原 `/data8t/Users/jiayl/neural_rx/` 工程及其环境未修改。

当前项目独立 Conda 环境为 `./.conda/fm-receiver`，包含 Python 3.11.15、
PyTorch 2.11.0+cu128、Sionna-no-RT 2.2.0、h5py 3.16.0。
CUDA 运行时为 12.8，实际加载的 cuDNN 版本为 91900、NCCL 为 2.28.9。
`pip check` 通过；45 个 CUDA 12 共享库的哈希均与对应 wheel 的 RECORD 一致。
完整版本见 [requirements.lock.txt](requirements.lock.txt)，环境校验见
environment_integrity.json（本地 `results/environment_integrity.json`）。

## Git 环境交付与独立 Conda 验证

Git 中的环境入口为 [environment.yml](environment.yml)，Conda 基础包由
[conda-linux-64.lock](conda-linux-64.lock) 固定公开 URL、构建版本与 SHA256，
pip 依赖由 [requirements.lock.txt](requirements.lock.txt) 固定版本及来源。
安装器不导出个人环境，也不克隆已有环境；解释器、标准库和科研依赖均在新的项目目录中安装。
环境目录和缓存不提交 Git，其他人克隆源码后按 [README.md](README.md) 重建。

本会话网络受限，实际安装使用官方发行包的本地镜像及 wheel 缓存；Conda 包的 SHA256
与 Git 中的公开下载清单一致。已分别创建主项目环境和一份不含旧环境、缓存或结果的源码副本环境。
在线下载整条流程尚未端到端执行，安装后的环境独立性和软件运行已实际验证：

- 两个环境均满足 `sys.prefix == sys.base_prefix`，Python、标准库及科学计算库均从各自新目录加载；项目模块来自各自源码。
- 两份源码的测试均为 11 项中 10 项通过、1 项因 GPU 不可见而跳过。
- 源码副本从 `/tmp` 启动直通测试，16 TB 零错误、CRC 全通过。
- 主项目完整 CPU 运行见 b0_conda_final（本地 `results/b0_conda_final/`）：256 TB 直通、7 个 AWGN 点及 64 帧导出重放均通过；所有 AWGN 输入哈希和误码计数与迁移前一致。
- 重复安装未改变环境定义与锁文件；显式导出的锁文件保持公开下载地址及准确版本，没有作者的绝对路径。
- 安装器拒绝复用旧 `.venv` 等未由本项目管理的环境。Git 忽略规则排除环境、缓存、结果和根目录数据，同时保留 `src/fm_receiver/data/` 源码。
- Ruff 和 Bash 语法检查通过。

检查结果、环境路径与日志保存在 verification.json（本地 `results/conda_portability/verification.json`）。
每次新评估还会在 `environment/` 保存环境定义、双锁文件与 `pyproject.toml`，并记录其哈希。
下文的 `b0_cpu_verified`、`b0_cpu_reproduced` 是此前 `.venv` 的历史结果，
`b0_conda_final` 是本次独立 Conda 环境的验证结果。

## 生成器参数

生成器回读：每 TB 848 个信息比特，速率匹配／扰码后 1248 个 Mapper 比特；624 个数据 RE，
DMRS 符号索引 `[2]`。单码块，LDPC BG1，lifting factor 40。实际信息码率为 848/1248，
与目标码率 679/1024 分开记录。软件 FFT 为 48、CP 为 4 样本、采样率为 720 ksample/s；
这些数值仅描述软件链路，X410 参数仍待接口验证。

## CPU 验收结果

以下计数取自早期 CPU 结果 b0_cpu_verified（本地 `results/b0_cpu_verified/`），
本次独立 Conda 运行 b0_conda_final（本地 `results/b0_conda_final/`） 得到相同计数。
无噪声直通测试 256 个 TB、217088 个信息比特，逐位零错误、错误块数为 0，CRC 全部通过。

AWGN 使用数据符号 Es/N0，以下各点独立设置随机种子；BER、BLER 在译码后统计。

| Es/N0 (dB) | 错误比特 / 总比特 | 错误 TB / 总 TB | CRC 失败 | BER | BLER |
|---|---|---|---:|---:|---:|
| -4 | 70963 / 217088 | 256 / 256 | 256 | 0.326886 | 1 |
| -2 | 59252 / 217088 | 256 / 256 | 256 | 0.272940 | 1 |
| 0 | 46535 / 217088 | 256 / 256 | 256 | 0.214360 | 1 |
| 2 | 31803 / 217088 | 256 / 256 | 256 | 0.146498 | 1 |
| 4 | 6409 / 217088 | 150 / 256 | 150 | 0.029523 | 0.585938 |
| 6 | 0 / 3473408 | 0 / 4096 | 0 | 0（实测） | 0（实测） |
| 8 | 0 / 3473408 | 0 / 4096 | 0 | 0（实测） | 0（实测） |

误码趋势随 Es/N0 提高而下降。零错误仅表示当前样本未观察到错误；曲线中的 `1/N`
三角标记用于显示，不代表误码率估计或置信上界。

- 完整计数表（本地 `results/b0_cpu_verified/metrics.csv`）、BER／BLER 曲线（本地 `results/b0_cpu_verified/ber_bler.png`）
- 最终生成配置（本地 `results/b0_cpu_verified/resolved_config.json`）、环境与设备（本地 `results/b0_cpu_verified/environment.json`）
- 检查结果（本地 `results/b0_cpu_verified/checks.json`）、运行日志（本地 `results/b0_cpu_verified/run.log`）

默认接收机对照在直通及每个 AWGN 点的首批观测上通过，恢复比特和 CRC 完全一致。
OFDM 调制／解调往返检查通过；初批最大栅格差约 `4.2e-7`，恢复比特正确、CRC 全通过。
实测噪声总功率及实虚部功率与设定方差一致。

## 复现、接口及测试

```bash
conda activate ./.conda/fm-receiver
python scripts/evaluate.py --device cpu --output results/cpu_first
python scripts/evaluate.py --device cpu --output results/cpu_repeat --compare-run results/cpu_first
```

历史重复运行结果保存在 b0_cpu_reproduced（本地 `results/b0_cpu_reproduced/`）。两次运行的生效配置和源码哈希一致，
每个 AWGN 点的完整输入 SHA256、比特／块错误计数及 CRC 失败数完全相同，复现比较通过。
历史结果目录已存在；上面的命令展示当前 Conda 环境的复现方法，应使用全新的输出目录。

独立种子 43、2 dB 下导出 64 帧，保存后重新读取、重新调制比对并重放接收机均通过：
NPZ 样本（本地 `results/b0_cpu_verified/samples/samples.npz`）、
字段与维度（本地 `results/b0_cpu_verified/samples/samples.json`）、
重放检查（本地 `results/b0_cpu_verified/samples/checks.json`）。标签来自实际 Mapper 输入，保留原始 TB 比特。

工具会话中的 `unittest` 共 11 项：10 项通过，CPU／GPU 同观测比较因该会话未开放 GPU 而跳过。
覆盖配置错误、CRC 独立计数、噪声功率错误检测、直通、默认接收机对照、映射与 LLR 符号、
OFDM 往返、种子复现、非整批样本导出和重放。日志见 unit_tests.log（本地 `results/unit_tests.log`）。
Ruff 和 Bash 语法检查通过；从 `/tmp` 启动 Hello World 也通过，入口不依赖当前工作目录。

## 独立 Conda 环境 GPU 验收与重复运行

用户已在普通 SSH 终端使用项目独立 Conda 解释器完成两次完整默认配置的 GPU 运行，结果为
第一次（本地 `results/20261116T091432103629Z_evaluate/`） 与
第二次（本地 `results/20261116T104302933134Z_evaluate/`）。
环境记录确认实际执行设备为 `cuda:0`、NVIDIA GeForce RTX 3090 24GB，驱动 570.133.07。
两次的 `executable` 均为项目 `.conda/fm-receiver/bin/python`，
`prefix == base_prefix` 均指向项目 `.conda/fm-receiver`，不依赖个人环境的解释器。
源码及四份环境定义的哈希与当前项目一致，依赖检查通过，运行状态为 `completed`，
`gpu_validation_passed=true`、`milestone_accepted=true`。

- 无噪声直通 256 TB、217088 个信息比特零错误，CRC 全通过。
- 默认接收机对照、Mapper 标签对齐、OFDM 往返、同观测 CPU／GPU 对照和固定种子复现均通过。
- 独立种子导出并重放 64 帧，映射、保存后重放和帧标识唯一性检查均通过。

| Es/N0 (dB) | 错误比特 / 总比特 | 错误 TB / 总 TB | CRC 失败 | BER | BLER |
|---|---|---|---:|---:|---:|
| -4 | 70978 / 217088 | 256 / 256 | 256 | 0.326955 | 1 |
| -2 | 59255 / 217088 | 256 / 256 | 256 | 0.272954 | 1 |
| 0 | 46051 / 217088 | 256 / 256 | 256 | 0.212131 | 1 |
| 2 | 31980 / 217088 | 256 / 256 | 256 | 0.147314 | 1 |
| 4 | 6061 / 217088 | 139 / 256 | 139 | 0.027920 | 0.542969 |
| 6 | 0 / 3473408 | 0 / 4096 | 0 | 0（实测） | 0（实测） |
| 8 | 0 / 3473408 | 0 / 4096 | 0 | 0（实测） | 0（实测） |

CPU 和 GPU 各自产生的随机噪声不同，不要求两设备 AWGN 输入哈希或错误计数逐位相同；
同一批观测上的 CPU／GPU 对照已通过。

- 运行与验收状态（本地 `results/20261116T104302933134Z_evaluate/manifest.json`）、实际环境（本地 `results/20261116T104302933134Z_evaluate/environment.json`）
- 完整计数（本地 `results/20261116T104302933134Z_evaluate/metrics.csv`）、曲线（本地 `results/20261116T104302933134Z_evaluate/ber_bler.png`）、检查结果（本地 `results/20261116T104302933134Z_evaluate/checks.json`）

已离线核对两次运行：生效配置、源码哈希、全部 AWGN 输入 SHA256、比特／块错误计数、
总比特／块数和 CRC 失败数均相同。核对结果保存在
b0_conda_gpu_acceptance.json（本地 `results/b0_conda_gpu_acceptance.json`）。

早期 20261116T085446012979Z_evaluate（本地 `results/20261116T085446012979Z_evaluate/`）
使用旧 `.venv`，其基础解释器来自个人 `deepreceiver`；该记录仅保留为历史 GPU 运行。
上述两次独立 Conda 运行已补齐环境隔离要求，首轮 B0 验收完成，无需再补跑原有 AWGN 验收。

后续运行可继续在普通 SSH 终端显式指定项目解释器：

```bash
./.conda/fm-receiver/bin/python -I -m unittest discover -s tests -t . -v
./.conda/fm-receiver/bin/python -I scripts/evaluate.py --device cuda:0
```

新运行会建立新的时间戳结果目录。CPU 运行的 `milestone_accepted` 仍按程序定义为 False；
实际 GPU 验收以此节的独立 Conda 运行及环境核对为准。

后续顺序为固定 TDL-A、Transformer＋相同 TB 译码器、G-ECCT、X410 实测入口。

## 后续实现的 CPU 验证

本轮接入 TDL-A、B1、G-ECCT/B2 的实现与训练／调度接口，未进行正式 GPU 模型训练。

- TDL-A 完整 CPU 验证（本地 `results/b0_tdl_cpu_verified/manifest.json`）完成：理想 CSI 无噪声恢复、
  信道归一化、噪声功率、NR 编码 oracle 以及 9 点扫描、64 帧导出和重放均通过。
- TDL-A 的 BER/BLER 从 −4 dB 的约 `0.3314/1` 下降到 12 dB 的约 `0.00299/0.02075`。
  完整计数（本地 `results/b0_tdl_cpu_verified/baseline/metrics.csv`）和
  诊断（本地 `results/b0_tdl_cpu_verified/diagnostics.json`）保留实际配置与错误计数；不要求衰落信道无误码。
- 当前 NR 适配覆盖固定单码块 RV0：848 TB 信息比特，含 CRC 后 864 比特，
  2720 位母码保留 1344 位；校验矩阵为 `464×1344`，显式处理 80 个穿孔位和 16 个零位。
- 训练检查覆盖联合网络非零梯度、microbatch 划分不改变输入，以及检查点续训与连续优化结果一致。
- 最终测试日志（本地 `results/implementation_checks/unit_tests.log`）为 17 项中 16 项通过、1 项 GPU 检查跳过；
  Ruff、Bash 语法和依赖检查通过。`results/research_smoke_final/` 另外完成 outer、decoder、joint
  各 2 步及每点 2 TB 的同观测评估，仅用于验证训练、保存、加载和评估代码路径。
- 离线 IQ 检查（本地 `results/x410_offline_first/checks.json`）通过样本偏移、频偏补偿、缺帧剔除、
  FFT 与 TB/CRC 重放；不读取监督标签文件。未操作实际 X410，也未验证盲同步。

源码实现不等于模型收敛。正式过拟合门限、B0/B1/B2 GPU 对比和远端部署仍需后续执行；
用户要求本轮收尾后暂停。恢复入口与目录约定见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

任务恢复后，用户已在普通 SSH 完成 602-2 的认证和硬件预检，确认远端主机名 `amax`、三张
RTX 3090，连同 602-1 共四个不同 GPU UUID；本地解释器指向项目 `.conda/fm-receiver`。
远端项目环境仍待部署，不能用硬件预检的 `ready=true` 代替其环境验收。
最新 22 项测试中 21 项通过、1 项 GPU 检查跳过，日志见
unit_tests_connection_resume.log（本地 `results/implementation_checks/unit_tests_connection_resume.log`）。
新增检查覆盖采样率转换、非法帧索引、事件迭代器、预检失败分类和不提交训练的部署入口。

随后 远端环境准备（本地 `results/worker_prepare/20261116T132326/state.json`）已完成，整体及两个 worker
状态均为 `ready`。两端 `prefix == base_prefix` 均指向各自项目目录下的独立 Conda 环境，
PyTorch 2.11.0+cu128、CUDA 12.8，源码一致检查均为 True。602-1 一张、602-2 三张 RTX 3090
的 UUID 不重复。该准备过程没有启动训练；正式 GPU 训练的验收仍待流水线执行。

## 多机 GPU 流水线与过拟合门限续跑

正式运行位于 20261116T134137（本地 `results/pipelines/20261116T134137/state.json`），两端独立环境和四个
GPU UUID 已通过检查。TDL-A 的 GPU manifest（本地 `results/pipelines/20261116T134137/jobs/tdl/output/manifest.json`）
为 `completed`、`gpu_accepted=true`。神经网络阶段当前进度如下：

| 固定 64 TB 门限 | 通过步数 | 信息比特错误 | 错误块 | CRC 失败 |
| --- | ---: | ---: | ---: | ---: |
| B1，种子 42 | 950 | 0 | 0 | 0 |
| B1，种子 43 | 1000 | 0 | 0 | 0 |
| G-ECCT，种子 42 | 1250 | 0 | 0 | 0 |
| G-ECCT，种子 43 | 1150 | 0 | 0 | 0 |

G-ECCT 首次检查在 1000 步未通过，因此原主控阻止了后续依赖。
本次沿用原 GPU、检查点、数据和优化器，将最大检查步数延长到 4000，实际在上述步数通过，
没有修改门限或冻结源码。原失败记录保存于运行目录 `recovery/before/`。
续跑仅在核对真实零错误结果后接纳门限，并自动恢复原流水线，状态为 `pipeline_resumed`；
验收记录（本地 `results/pipelines/20261116T134137/recovery/acceptance.json`）保存步数、模型哈希和主控信息。

B1 种子 42 完成 20000 步，最佳检查点在 19250 步；种子 43 按既定规则在 16250 步早停，
最佳检查点在 14250 步。两份最佳权重的源码与种子已核验。
两个 G-ECCT 正式训练已在 602-2 启动，后续为对应种子的联合微调和独立同观测评估。
这些门限和训练状态尚不构成性能提升结论；正式 B0/B1/B2 对比仍待完成。

## Full access 下的 GPU 回归检查

用户开放权限后，工具已直接验证本机 RTX 3090、602-2 SSH 及远端三张 GPU 可用，
并复核两端独立 Conda 前缀和冻结源码一致性，记录为
full_access_verification.json（本地 `results/implementation_checks/full_access_verification.json`）。
完整测试本次为 **22 项全部通过、零跳过**，包含此前受限而跳过的同观测 CPU/GPU 对照，
日志为 unit_tests_full_access_gpu.log（本地 `results/implementation_checks/unit_tests_full_access_gpu.log`）。
没有修改运行中的模型源码、配置、依赖或门限。

## 正式多机流水线最终结果

流水线状态（本地 `results/pipelines/20261116T134137/state.json`）最终为 `completed`，14 个任务全部结束。
两个种子的正式训练及联合微调完成，按既定规则早停的任务也均正常保存最佳检查点。
[报告验收](reports/2026-11-16/acceptance.json)核对 108 行结果、模型 SHA256、
冻结源码、每点同观测哈希和样本数，以及 100 错误块/4096 TB 的停止条件。

| 场景 | TB 数/种子 | B0 BLER | B1 BLER（42 / 43） | B2 BLER（42 / 43） |
| --- | ---: | ---: | --- | --- |
| AWGN，4 dB | 4096 | 0.561035 | 0.020508 / 0.020508 | 1 / 1 |
| TDL-A，12 dB | 4096 | 0.019775 | 0.008789 / 0.009277 | 0.250977 / 0.246582 |

每个种子的 18 个点中，B1 相对 B0 有 6 点更低、12 点相同；B2 相对 B1 无更低点、8 点相同、
10 点更高。以上为实测结果，不宣称统计显著性。完整表格、CSV、PNG/PDF 见
[正式报告](reports/2026-11-16/README.md)。

[补充诊断](reports/2026-11-16/diagnostics/NOTES.md)使用独立 16 TB 和真值辅助 oracle，
未改变正式指标或选取模型。理想 Mapper 输入能正确恢复；带噪 LLR 下仍有残余错误，
真值填回穿孔位也未全部消除错误。具体退化原因仍需后续受控消融，不将负面结果写成提升。
本轮冻结源码和锁文件保存在运行目录 `frozen_source.tar.gz`，便于后续实验变更后追溯。
