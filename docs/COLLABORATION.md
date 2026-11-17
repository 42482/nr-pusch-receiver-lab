# 同伴拉取与复现

仓库名为 `nr-pusch-receiver-lab`，默认创建私有仓库。代码、示例配置、环境锁文件、测试、
文档及小型实验结果随 Git 分发；环境、缓存、权重、原始数据、历史对话和本地 worker 配置留在本地。

```bash
git clone <GitHub 仓库地址> nr-pusch-receiver-lab
cd nr-pusch-receiver-lab
bash scripts/setup_env.sh
conda activate ./.conda/fm-receiver
python -m unittest discover -s tests -t . -v
python scripts/evaluate.py --device cuda:0
```

先安装并初始化 Conda。支持 Linux x86_64，锁定 Python 3.11.15、Torch 2.11.0+cu128、
Sionna 2.2.0；GPU 机器需提供兼容的 NVIDIA 驱动。CPU 检查可指定 `--device cpu`。
每份 checkout 通过锁文件建立自己的 `.conda/fm-receiver`，不依赖作者的个人环境。

默认单机运行不需要复制作者的 worker 地址。多机时复制 `config/workers.example.toml` 为
`config/workers.local.toml`，填自己的地址、数据盘目录及 GPU 数；该文件不提交 Git。
所有 checkout、环境和缓存应放在自己的数据盘，生成数据根目录默认是 checkout 的上级目录。

## 版本与结果

- `frozen-20261116`：原始已完成实验的核心源码，其每个源码/锁文件哈希已与运行记录核对。
- `v0.1.1` / `main`：发布前复查后的版本，修复比较任务重启校验、主控启动确认，固定帧检查上限为 4000 步。
- [共享结果](../reports/2026-11-16/README.md)对应冻结版本，不因后续修复改写原始计数。

旧检查点校验冻结源码；复用旧权重时使用对应版本，不能静默替换来源哈希。
新增实验使用新的配置和输出目录，保持模型、训练、接收适配、评估、调度与硬件前端的模块边界。

## 首次上传

在本项目目录执行：

```bash
bash scripts/publish_github.sh
```

脚本在需要时从官方 release 下载并校验 GitHub CLI，保存在本项目 `.cache/tools/gh`。
首次运行按提示在浏览器登录 GitHub；脚本使用实际授权的账号，创建私有仓库并推送主分支和版本标签。
认证配置放在本项目 `.cache/gh`，不写入 Git。默认不修改全局 Git 认证配置，不覆盖不同的 origin，
不强制推送。上传后在仓库 Settings → Collaborators 添加同伴，获邀账号即可拉取。
