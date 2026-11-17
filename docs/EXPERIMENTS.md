# TDL-A、B1、G-ECCT/B2 的运行与恢复

## 当前状态

正式运行 `results/pipelines/20261116T134137/` 已全部完成，14 个任务为 `completed`。
TDL-A GPU 验收及 B1/G-ECCT 两种子的固定帧门限通过，独立训练、联合微调和两种子的
正式同观测评估均已结束。报告含 108 行计数，来源和停止条件已核验。

[正式报告](../reports/2026-11-16/README.md)显示 B1 在有效比较点优于 B0，
当前 B2 没有提升；[补充诊断](../reports/2026-11-16/diagnostics/NOTES.md)保留负面结果及待查方向。
工具现有 full access，已直接验证 CUDA 和 SSH；22 项完整测试全部通过，GPU 项零跳过。
此次任务无需再次执行 `start`、`resume` 或 `recovery/start.sh`。

以下命令供新实验参考。首轮 G-ECCT 在默认 1000 步未通过固定帧检查，续训到 1250/1150 步才通过；
`v0.1.1` 流水线已将过拟合检查上限设为 4000 步，原门限保持不变、通过即停。
独立检查可使用 `train.py --stage decoder --overfit --steps 4000`，已有检查点加 `--resume`。
新增源码版本应保留本轮 `frozen_source.tar.gz` 和模型来源，不覆盖原输出。

## 环境与目录

每台机器使用自己项目目录内的 `.conda/fm-receiver`。主控为 602-1，远端三卡机器为
602-2；具体地址在被 Git 忽略的 `config/workers.local.toml`。可参考
`config/workers.example.toml` 配置其他机器。密码不写入文件；启动前由系统 SSH 交互认证，
使用位于数据盘的连接复用 socket。连接中断后，已启动的远端 Screen 任务仍由远端托管；
若连接复用失效，重新执行 `connect` 后再恢复主控。

所有生成文件留在 `/data8t/Users/jiayl/`：

- `Receiver/.conda`、`.cache`：解释器、下载／编译缓存、临时目录、Screen 和 SSH socket。
- `Receiver/results/pipelines/<run>/`：主控状态、日志、每个任务及训练检查点。
- `Receiver-worker/`：远端独立代码、环境与运行结果。已有的非本项目目录不会被覆盖。

不修改 `HOME`。换用户或盘符时设置 `FM_RECEIVER_DATA_ROOT` 并修改本地 worker 配置；
环境锁文件不包含作者的解释器路径。

## 从普通 SSH 启动

先完成环境安装。以下命令在能实际访问 GPU 的普通 SSH 终端执行：

```bash
cd /data8t/Users/jiayl/Receiver
bash scripts/run_pipeline.sh connect
bash scripts/run_pipeline.sh preflight
bash scripts/run_pipeline.sh prepare
bash scripts/run_pipeline.sh start
```

`start` 会先建立远端 SSH 复用连接，然后启动主控 Screen。主控在后台完成远端环境安装、
源码哈希检查和 GPU 预检，使用 GPU UUID 分配独立任务。不会把两台服务器混成跨机 DDP。
预期资源是 602-1 一张、602-2 三张 3090；不足、重复 GPU UUID 或环境不匹配会明确报告。
远端预检失败时，本地可用 GPU 可继续承担任务，状态文件记录实际使用的资源。

`prepare` 仅同步代码、安装远端独立环境并核对两端源码及解释器，不启动训练。
日志和状态写入 `results/worker_prepare/<时间>/`；远端安装输出会同步写入本地日志，便于检查进度。
普通 `preflight` 在远端环境尚未部署时只检查连接和 GPU；`prepare` 成功才确认远端环境及源码一致。
Conda 安装关闭 home 下的环境注册，并将包缓存固定到数据盘；已有 Conda 仅作为包管理器使用。

默认墙钟上限 24 小时，单个任务上限 8 小时。改变整个运行预算可用 `--max-hours`。
只有收到主控心跳才会确认提交；Screen 返回零退出码本身不代表后台任务已开始。

```bash
bash scripts/run_pipeline.sh status
bash scripts/run_pipeline.sh status --run results/pipelines/<run>
bash scripts/run_pipeline.sh connect
bash scripts/run_pipeline.sh resume --run results/pipelines/<run>
```

启动输出会给出带 `SCREENDIR` 的重新连接命令。在 Screen 中按 `Ctrl-a`，再按 `d`，
可脱离终端而保留其进程。服务器重启、进程失败或预算到期需要依靠检查点恢复，Screen 不负责恢复模型。
恢复要求相同源码、编码布局、数据和优化器配置；活跃主控不允许重复恢复。

**这里托管的是实验流水线，不会把已经运行在前台 SSH 中的 Codex 会话迁入 Screen。**
需要保持 Codex 时，应在当前对话收尾后另开 Screen，再从其中恢复 Codex 会话。

## 任务与验收门限

1. `tdl`：100 ns 时延扩展、3.5 GHz、3 m/s、归一化 TDL-A。检查 CP、噪声、理想 CSI、
   编码 oracle、种子复现及保存后重放，再跑 −4 到 12 dB 的完整扫描。
2. `overfit_outer/decoder`：固定 64 帧检查。外接收机需 Mapper BER ≤ 1e-3 且 TB 全部正确；
   G-ECCT 需 TB 全部正确。未通过时不启动依赖该模型的正式训练。
3. `train_outer/decoder`：种子 42、43 的独立任务。B1 训练 AWGN/TDL-A 混合，G-ECCT
   独立训练使用 AWGN 传统软检测 LLR。数据流与固定验证集分开，测试数据不用于模型选择。
4. `joint`：加载对应种子的两个正式最佳检查点，联合微调 B2，持续检查两段网络的梯度。
5. `compare`：在相同的独立接收观测上评估 B0/B1/B2。每点至少 256 TB，所有接收机均累计
   100 个错误 TB 后可停止，否则最多 4096 TB。零错误仅表示本次未观察到错误。

训练默认 2 层、128 维、8 头、FFN 512、Pre-LN、dropout 0；AdamW，权重衰减 1e-4、
梯度裁剪 1、有效 batch 16。独立训练学习率 1e-3、最多 20000 步，联合训练 1e-4、最多
5000 步。每 250 步验证，连续 8 次无改善早停。前向／反向 OOM 可自动缩小 microbatch，
相同有效 batch 的发送比特和观测保持一致。

`last.pt` 保存网络、优化器、随机状态、有效步数和数据坐标，`best.pt` 保存最佳验证模型。
输出还包括配置、源码哈希、环境、逐次验证日志和验收状态。未达到门限、预算用尽或失败
都不会被记作正式验收通过。

## 单独诊断

```bash
./.conda/fm-receiver/bin/python -I scripts/validate_tdl.py --device cuda:0 --output results/tdl_new
./.conda/fm-receiver/bin/python -I scripts/train.py --stage outer --overfit --output results/outer_check
./.conda/fm-receiver/bin/python -I scripts/train.py --stage decoder --overfit --output results/decoder_check
./.conda/fm-receiver/bin/python -I scripts/x410_offline.py --output results/offline_new
```

训练可用 `--resume`；联合训练另外指定 `--outer-checkpoint` 和 `--inner-checkpoint`。
流水线 `--smoke` 只做极短的基础设施诊断，跳过正式过拟合门限；最终状态为
`diagnostic_completed`，不能用作模型训练验收。
