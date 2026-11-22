# 置信度感知自适应情感分析（MR / SST-2）

使用 PyTorch 做英文正负情感分类。直接读取 SentiCSE 的 `mr/prepared_seed42` 和 `sst2/prepared_seed42` 下的 `train.csv`、`validation.csv`、`test.csv`，不重分数据。标签 `0=negative`，`1=positive`。

| 数据 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| MR | 8,530 | 1,066 | 1,066 |
| SST-2 | 6,909 | 872 | 1,821 |

这是直接监督的文本分类。参考工程的本地分割协议被保留，SentiCSE 的四元组对比学习、情感 MLM 和 Spearman 选模不用于此任务。MR 是本地分层划分，SST-2 是本地整理版；结果应和使用同一划分的实验比较。

## 运行

Python 3.10+。现有环境已安装依赖时，可跳过安装。Windows 用 `python` 或实际的 Python 可执行文件；服务器可通过 `--data-root` 改数据位置。

```powershell
python -m pip install -r requirements.txt
python train.py --dataset all --encoder scratch --epochs 8 --output-dir runs_offline
```

`--dataset mr` 或 `--dataset sst2` 可单独训练。`--device auto` 自动选择 CUDA 或 CPU。离线模式的词表仅取训练集；编码器为随机初始化的小型双向 Transformer（BERT 风格），不使用外部预训练权重。若要运行 PDF 描述的预训练轻量 BERT 方案：

```powershell
python -m pip install -r requirements-pretrained.txt
python train.py --dataset all --encoder pretrained --model-name-or-path 'C:\models\distilbert-base-uncased' --epochs 8 --batch-size 32 --output-dir runs_pretrained
```

请把示例模型目录改为实际含权重和 tokenizer 的本地目录。也可以指定 `--model-name-or-path distilbert/distilbert-base-uncased --allow-download` 下载模型。预训练编码器以 `--encoder-lr` 微调；其余层用 `--lr`。

训练和验证都使用 `tqdm` 显示进度、累计 loss 和 accuracy。每轮更新各数据集下的 `loss_curve.png` 和 `history.csv`。默认输出位置是项目内 `runs/`；如果已有同名数据集的实验，请给新实验指定新的 `--output-dir`。

```powershell
python predict.py --run-dir runs/mr --text 'A lovely and moving film.' --text 'Boring and predictable.'
```

随项目保存的 `runs/` 是两套数据用 `--encoder scratch --epochs 6 --device cpu --seed 42` 的实跑结果。图中的两条曲线都是总损失；训练曲线是在逐步更新参数时累计的，验证曲线是每轮结束后用固定权重、关闭 dropout 算出的。

## 代码和路由

| 文件 | 负责内容 |
| --- | --- |
| `data.py` | 读取三份 CSV，分词和动态补齐 |
| `model.py` | TextCNN、降维 TSSA、1D CAA、动态路由与损失 |
| `train.py` | 训练、验证选模、测试、曲线和结果记录 |
| `predict.py` | 恢复模型并输出标签、置信度和路由 |

正文先编码为 `[B, L, D]`。TextCNN 使用固定大小 `3/5/7` 的一维卷积核沿 token 轴提取不同跨度的特征。深层分支从编码器输出降维 `D → trans_dim`，使用模块 129 的 TSSA（增加 padding mask）和残差、前馈网络，再升回 `D`。TSSA 保留按头做 softmax、沿有效 token 计算二阶矩的做法，见 [ToST 官方实现](https://github.com/RobinWu218/ToST)。CAA 源自模块 083，改为一维局部平均和串联深度卷积，用逐 token、逐通道的 sigmoid 权重融合两路表示。三个分支共用分类头。所有池化、统计和卷积边界处理均考虑 padding。

置信度取二分类 logits 的最大 **softmax** 概率。推理规则逐样本执行：

1. `C_cnn >= tau`：从 CNN 提前退出，不执行 TSSA 或 CAA。
2. `C_cnn < tau`：执行 TSSA。若 `C_trans >= C_cnn`，用 TSSA 结果。
3. 其余样本执行 1D CAA，用融合结果。

`tau` 默认 `0.8`。训练用硬路由选出主任务 CE，但计算全部分支，通过三路辅助 CE 确保各分支收到监督；额外的多类 Brier 损失直接惩罚概率与真实类别向量的偏差。损失为 `主任务 CE + 0.4 × 三路平均辅助 CE + 0.1 × 三路平均 Brier`（可通过参数调整）。Brier 是校准目标，不能据此假定两个分支已充分校准。验证 loss 与训练使用同一目标，测试用真正的稀疏推理。更大的 `tau` 通常会让更多样本进入深层；比较阈值要用验证集，而非测试集。

训练产物还包括 `best_model.pt`（按验证准确率选模，同分用验证 loss）、`config.json`、`test_metrics.json`（准确率、Macro-F1、混淆矩阵和三条路径的覆盖率/准确率），以及词表或 tokenizer/编码器配置。TSSA 算子本身不构造 `L × L` 注意力矩阵；前置 BERT 类编码器仍有常规注意力开销，整体速度需要实测。
