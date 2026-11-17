# 当前实施接续记录

**当前状态：原定软件任务已完成并核验，流水线 14 个任务全部为 `completed`，两个种子的 108 行正式同观测结果已汇总。B1 在本轮有效比较点优于 B0；当前 G-ECCT/B2 没有性能增益，负面结果已保留。full access 已验证，后续终端操作可由工具执行，目前不需要用户手动输入命令。**

## 结果入口

- [正式对比报告](reports/2026-11-16/README.md)、[完整计数](reports/2026-11-16/metrics_all.csv)、[BLER 图](reports/2026-11-16/bler_all.png)。
- 流程状态（本地 `results/pipelines/20261116T134137/state.json`）和[报告验收](reports/2026-11-16/acceptance.json)。
- [G-ECCT 补充诊断](reports/2026-11-16/diagnostics/NOTES.md)。
- [X410 接口与准备清单](docs/X410_INTERFACE.md)：离线 IQ 适配已验证，尚未射频实采。
- 原始核心源码已保存为 Git 标签 `frozen-20261116`；本轮冻结源码、锁文件和测试另存于运行目录 `frozen_source.tar.gz`，文件哈希见 `frozen_source_manifest.json`；不含个人环境、缓存或 SSH 认证信息。

## 已完成的验证

- B0 直通/AWGN 和固定 TDL-A 的 CPU/GPU 验收通过，TDL GPU manifest 的 `gpu_accepted=true`。
- B1 两种子在 950/1000 步通过固定 64 TB 门限。G-ECCT 初次 1000 步失败，原检查点续训至 1250/1150 步后均为信息比特零错误、块零错误、CRC 全通过；未放宽门限。
- 原失败记录保存在 `recovery/before/`，续跑证据为 `recovery/acceptance.json`。仅预算延长，原数据、优化器、模型和源码保持不变；流水线沿用原 24 小时预算的剩余时间。
- 两种子的独立训练、联合微调和正式评估全部结束。提前停止来自既定验证规则，不是运行异常。
- 每个种子的 18 个场景/SNR 组合内，B0/B1/B2 的输入 SHA256 和 TB 数相同。每点 256–4096 TB；不足 4096 时已核对所有接收机达到 100 错误块的停止条件。
- 22 项测试全部通过、GPU 项零跳过，日志为 `results/implementation_checks/unit_tests_full_access_gpu.log`。原有 Ruff、Bash 语法和依赖检查通过；本次新增报告/诊断脚本 Ruff 通过。
- NR 校验子、Mapper 排列、TB/CRC oracle、联合梯度、microbatch 输入一致性和训练检查点恢复均已验证。
- X410 模拟 IQ 的频偏、帧偏移、缺帧剔除、FFT 和 CRC 重放通过；0.96/1.44/7.68 Msample/s 回到 720 ksample/s 的模拟采样率转换通过。不能据此宣称真实 RF 链路或盲同步通过。

## 训练记录

| 任务 | 最后步数 | 最佳步数 | 训练终止状态 |
| --- | ---: | ---: | --- |
| train_outer_42 | 20000 | 19250 | completed |
| train_decoder_42 | 9250 | 7250 | early_stopped |
| joint_42 | 2750 | 750 | early_stopped |
| train_outer_43 | 16250 | 14250 | early_stopped |
| train_decoder_43 | 7500 | 5500 | early_stopped |
| joint_43 | 5000 | 4000 | completed |

## 本轮性能结论

- AWGN 4 dB，两种子的 B0 BLER 均为 0.561035，B1 均为 0.020508，每种子 4096 TB。
- TDL-A 12 dB，B0 为 0.019775，B1 两种子为 0.008789/0.009277，均为 4096 TB。
- 每个种子的 18 点中，B1 相对 B0 有 6 点 BLER 更低、12 点相同；B2 相对 B1 无更低点、8 点相同、10 点更高。仅报告实测，不宣称统计显著性。
- 小样本诊断中理想 Mapper LLR 能正确恢复，带噪 LLR 仍有残余错误；真值填回穿孔位也未全部消除错误。尚未确定具体原因，不把退化直接归因于单一穿孔位映射问题。

## 环境与资源

- 602-1 一张、602-2 三张 RTX 3090，四个 UUID 不同；远端系统主机名 `amax`。
- 各自使用项目 `.conda/fm-receiver`，`prefix == base_prefix`，Python 3.11.15、Torch 2.11.0+cu128、Sionna 2.2.0；没有复用个人 Conda 解释器。
- 工具已直接验证 CUDA 和 SSH，证据为 `results/implementation_checks/full_access_verification.json`。此前 socket/GPU 受限属于历史状态。
- 新文件、环境、缓存和结果均留在 `/data8t/Users/jiayl/`；连接配置在 Git 忽略的 `config/workers.local.toml`，不保存密码。
- 此次训练和报告 Screen 已正常结束。Codex 的 Screen 与训练 Screen 独立，不能混同二者的生命周期。

## 后续接续

1. 先读本记录、正式报告和诊断，不要重复 `start`、`resume` 或启动已完成的恢复脚本。
2. 本轮配置保持为已完成的冻结实验。新增实验使用独立配置、输出目录和来源记录，沿用 `models`、`training`、`receivers`、`evaluation`、`hardware` 的模块边界。
3. 下一轮研究先针对 G-ECCT 的残余错误纠正、LLR 分布及泛化做受控消融，不在既有独立测试集上选择新权重。
4. X410 继续依据接口清单核对设备、时钟、采样率、同步和事件记录；当前软件验证不等于已完成真实采集。

本轮固定帧检查曾因默认 1000 步不足而需要续训，复现时应按记录允许 decoder 检查到 4000 步。
发布前复查的 `v0.1.1` 已修复比较 `--resume` 的 JSON 序列规范化和恢复时旧主控 PID 的误确认，新增两个回归检查，最新共 24 项测试全部通过。
新版本流水线固定帧检查上限设为 4000 步，达到既有门限即提前结束；历史实验原始计数保持不变。
GitHub 发布入口为 `bash scripts/publish_github.sh`，默认创建私有的 `nr-pusch-receiver-lab`；首次需要浏览器账号授权，目前尚未上传。
后续若修改核心源码，应另立版本并保留本轮源码归档；检查点默认要求相同冻结源码，不应静默改写其来源哈希。
