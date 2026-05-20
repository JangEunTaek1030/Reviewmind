# ReviewMind: End-to-End NLP Sentiment Classification (English | 中文)

## 1) Project Overview
ReviewMind is an end-to-end Natural Language Processing (NLP) sentiment classification project built on a balanced IMDb movie review subset. The project walks through dataset preparation, text preprocessing, baseline model training, model comparison, result visualization, and a Streamlit demo for interactive predictions.

- Dataset size: **2,500 reviews**
- Class balance:
  - **1,250 negative**
  - **1,250 positive**
- Current best traditional ML baseline: **TF-IDF + Logistic Regression**
- v2.0 Transformer baseline: **DistilBERT**

### 中文说明
ReviewMind 是一个端到端的自然语言处理（NLP）情感分类项目，基于一个类别平衡的 IMDb 电影评论子集。该项目覆盖了数据准备、文本预处理、基线模型训练、模型对比、结果可视化，以及用于交互式预测的 Streamlit 演示应用。

- 数据集规模：**2,500 条评论**
- 类别分布：
  - **1,250 条负向评论**
  - **1,250 条正向评论**
- 当前最佳传统机器学习基线：**TF-IDF + Logistic Regression**
- v2.0 Transformer 基线：**DistilBERT**

---

## 2) Features
- End-to-end sentiment classification workflow
- Balanced IMDb subset preparation pipeline
- Traditional ML baseline training (TF-IDF + Logistic Regression)
- Multi-model comparison:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
- Visualization outputs for model comparison and confusion matrix
- Streamlit web app for real-time sentiment inference

### 中文说明
- 端到端情感分类流程
- 平衡 IMDb 子集数据准备流程
- 传统机器学习基线训练（TF-IDF + Logistic Regression）
- 多模型对比：
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
- 模型对比与混淆矩阵可视化输出
- 基于 Streamlit 的实时情感预测 Web 演示

---

## 3) Project Structure
```text
Reviewmind/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data_preprocessing.py
│   ├── prepare_imdb_dataset.py
│   ├── train_ml.py
│   ├── compare_models.py
│   ├── train_transformer.py
│   ├── visualize_results.py
│   └── run_pipeline.py
├── outputs/
│   └── figures/
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       └── model_comparison_results.csv
├── requirements.txt
├── LICENSE
└── README.md
```

### 中文说明
```text
Reviewmind/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data_preprocessing.py
│   ├── prepare_imdb_dataset.py
│   ├── train_ml.py
│   ├── compare_models.py
│   ├── train_transformer.py
│   ├── visualize_results.py
│   └── run_pipeline.py
├── outputs/
│   └── figures/
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       └── model_comparison_results.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 4) Quick Start (Windows PowerShell)
### 4.1 Clone the repository
```powershell
git clone <your-repo-url>
cd Reviewmind
```

### 4.2 Allow script execution for current PowerShell session only (safer)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4.3 Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4.4 Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.5 Prepare IMDb dataset subset
```powershell
python src/prepare_imdb_dataset.py
```

### 4.6 Train baseline model (TF-IDF + Logistic Regression)
```powershell
python src/train_ml.py
```

### 4.7 Compare multiple models
```powershell
python src/compare_models.py
```

### 4.8 Generate visualizations
```powershell
python src/visualize_results.py
```

### 4.9 Run the full local ML pipeline (recommended)
```powershell
python src/run_pipeline.py
```

### 4.10 Run Streamlit demo
```powershell
streamlit run app/streamlit_app.py
```

### 4.11 Train v2.0 Transformer baseline (DistilBERT)
```powershell
python src/train_transformer.py
```

### 4.12 Fast Transformer debug check (quick environment test)
```powershell
python src/train_transformer.py --debug
```

### 中文说明
#### 4.1 克隆仓库
```powershell
git clone <your-repo-url>
cd Reviewmind
```

#### 4.2 仅在当前 PowerShell 会话中临时放开脚本执行（更安全）
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### 4.3 创建并激活虚拟环境
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 4.4 安装依赖
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4.5 准备 IMDb 子集数据
```powershell
python src/prepare_imdb_dataset.py
```

#### 4.6 训练基线模型（TF-IDF + Logistic Regression）
```powershell
python src/train_ml.py
```

#### 4.7 进行多模型对比
```powershell
python src/compare_models.py
```

#### 4.8 生成可视化结果
```powershell
python src/visualize_results.py
```

#### 4.9 一键运行完整本地机器学习流水线（推荐）
```powershell
python src/run_pipeline.py
```

#### 4.10 启动 Streamlit 演示
```powershell
streamlit run app/streamlit_app.py
```

#### 4.11 训练 v2.0 Transformer 基线（DistilBERT）
```powershell
python src/train_transformer.py
```

#### 4.12 Transformer 快速调试检查（环境快速验证）
```powershell
python src/train_transformer.py --debug
```

---

## 5) Latest Model Comparison Results
| Model | Accuracy | F1-score |
|---|---:|---:|
| Multinomial Naive Bayes | 0.8140 | 0.8140 |
| Logistic Regression | **0.8400** | **0.8398** |
| Linear SVM | 0.8040 | 0.8039 |
| DistilBERT Transformer | **0.8920** | **0.8929** |

**Current best baseline:** DistilBERT Transformer.

### 中文说明
| 模型 | Accuracy | F1-score |
|---|---:|---:|
| Multinomial Naive Bayes | 0.8140 | 0.8140 |
| Logistic Regression | **0.8400** | **0.8398** |
| Linear SVM | 0.8040 | 0.8039 |
| DistilBERT Transformer | **0.8920** | **0.8929** |

**当前最佳基线模型：** DistilBERT Transformer。

---

## 6) v2.0 Transformer Baseline (DistilBERT)
In v1.0, ReviewMind focused on traditional ML pipelines (e.g., TF-IDF + Logistic Regression).  
In v2.0, we add a Transformer baseline using DistilBERT for stronger semantic modeling.

### Experiment Setup
| Item | Value |
|---|---|
| Model | DistilBERT Transformer |
| Dataset | 2,500 balanced IMDb reviews |
| Train/Test split | 80/20 |
| Epochs | 2 |
| Accuracy | 0.8920 |
| Precision | 0.8858 |
| Recall | 0.9000 |
| F1-score | 0.8929 |

### v1.0 vs v2.0 Baseline Comparison
| Baseline | Accuracy | F1-score |
|---|---:|---:|
| Multinomial Naive Bayes | 0.8140 | 0.8140 |
| Logistic Regression | 0.8400 | 0.8398 |
| Linear SVM | 0.8040 | 0.8039 |
| DistilBERT Transformer | **0.8920** | **0.8929** |

### 中文说明
在 v1.0 中，ReviewMind 主要采用传统机器学习流程（如 TF-IDF + Logistic Regression）。  
在 v2.0 中，项目新增了基于 DistilBERT 的 Transformer 基线，以获得更强的语义建模能力。

### 实验设置
| 项目 | 数值 |
|---|---|
| 模型 | DistilBERT Transformer |
| 数据集 | 2,500 条平衡 IMDb 评论 |
| 训练/测试划分 | 80/20 |
| 训练轮数 | 2 |
| Accuracy | 0.8920 |
| Precision | 0.8858 |
| Recall | 0.9000 |
| F1-score | 0.8929 |

### v1.0 与 v2.0 基线对比
| 基线模型 | Accuracy | F1-score |
|---|---:|---:|
| Multinomial Naive Bayes | 0.8140 | 0.8140 |
| Logistic Regression | 0.8400 | 0.8398 |
| Linear SVM | 0.8040 | 0.8039 |
| DistilBERT Transformer | **0.8920** | **0.8929** |

---

## 7) Visualizations
### Model Comparison
![Model Comparison](outputs/figures/model_comparison.png)

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix.png)

### 中文说明
### 模型对比图
![模型对比图](outputs/figures/model_comparison.png)

### 混淆矩阵
![混淆矩阵](outputs/figures/confusion_matrix.png)

---

## 8) Streamlit Demo Notes & Limitations
The Streamlit app (`app/streamlit_app.py`) loads the locally generated model file:
- `outputs/models/ml_baseline_model.pkl`

Known limitations for text sentiment prediction:
- Very short inputs may lack enough semantic clues.
- Numeric-only ratings (e.g., “10/10”, “2 stars”) can be ambiguous without descriptive text.
- Sarcasm is difficult for traditional bag-of-words style models.
- Ambiguous or mixed-sentiment reviews may be misclassified.
- Context-dependent expressions (domain, culture, irony, discourse context) may not be fully captured.

### 中文说明
Streamlit 应用（`app/streamlit_app.py`）会加载本地生成的模型文件：
- `outputs/models/ml_baseline_model.pkl`

文本情感预测的已知局限性：
- 输入过短时，语义信息可能不足。
- 纯数字评分（如“10/10”“2 stars”）缺少上下文，可能导致歧义。
- 反讽/讽刺表达对传统词袋类模型较难识别。
- 含混或情感混合的评论更容易误判。
- 强依赖语境的表达（领域、文化、反语、上下文）可能无法被充分建模。

---

## 9) Artifact Policy (Important)
Generated artifacts should remain local and must **not** be committed:
- `data/processed/*.csv`
- `outputs/models/*.pkl`
- `outputs/models/transformer_baseline/` (large Transformer checkpoints/artifacts)

These files are reproducible by running the pipeline scripts and are intentionally ignored by Git.

### 中文说明
以下生成产物应仅保留在本地，**不要提交到 Git**：
- `data/processed/*.csv`
- `outputs/models/*.pkl`
- `outputs/models/transformer_baseline/`（Transformer 检查点/产物体积较大）

这些文件可通过项目脚本重新生成，因此已被 Git 忽略。

---

## 10) Key Learnings
- Building a clean NLP workflow is as important as model selection.
- Balanced subsets help reduce misleading metrics caused by class imbalance.
- Logistic Regression remains a strong and interpretable baseline for TF-IDF text features.
- Model comparison and visualization make trade-offs clearer than single-metric reporting.
- Lightweight app deployment (Streamlit) is an effective way to present ML projects in portfolios.

### 中文说明
- 构建清晰、可复现的 NLP 流程与模型选择同样重要。
- 平衡数据子集有助于降低类别不平衡带来的指标偏差。
- 在 TF-IDF 特征下，Logistic Regression 依然是强且可解释的基线模型。
- 模型对比与可视化相比单一指标更能体现权衡关系。
- 使用 Streamlit 进行轻量部署，是展示机器学习项目作品集的高效方式。

---

## 11) Roadmap
### Completed
- End-to-end IMDb sentiment classification pipeline
- Balanced 2,500-sample dataset preparation
- Baseline training with TF-IDF + Logistic Regression
- Traditional ML model comparison (NB / LR / Linear SVM)
- Visualization outputs (model comparison + confusion matrix)
- Streamlit interactive demo
- Explored transformer-based baselines for performance benchmarking

### Next Steps
- Add experiment tracking (e.g., MLflow or lightweight logging)
- Introduce cross-validation and hyperparameter tuning workflow
- Expand evaluation with error analysis by review length and linguistic patterns

### 中文说明
### 已完成
- IMDb 情感分类端到端流程
- 2,500 条平衡子集数据准备
- TF-IDF + Logistic Regression 基线训练
- 传统机器学习模型对比（NB / LR / Linear SVM）
- 可视化输出（模型对比图 + 混淆矩阵）
- Streamlit 交互式演示
- 已探索基于 Transformer 的基线模型并完成性能对标

### 下一步
- 增加实验跟踪（如 MLflow 或轻量日志方案）
- 引入交叉验证与超参数调优流程
- 扩展评估：按评论长度和语言模式进行误差分析
