# 信用风险 PD 模型验证与压力测试项目

> Graduate Quantitative Modeling Extension Project  
> Credit Risk PD Model Validation and Stress Testing

## 1. 项目背景与动机

本项目是一个研究生阶段量化建模拓展项目，目标是模拟金融机构模型验证分析师对信用风险 PD（Probability of Default，违约概率）模型进行独立验证的完整流程。

我此前的经历主要集中在机器学习模型、数学建模和模型评估。例如，在“华为杯”研究生数学建模竞赛中，我曾围绕区域双碳目标与路径规划开展预测建模、多情景分析和模型局限性评价；在算法实习和工作经历中，也接触过模型训练、模型选型、自动化测试和结果评估。

在进一步了解模型风险管理（Model Risk Management, MRM）后，我发现模型验证关注的不只是“模型能否预测”，还包括：

- 模型假设是否合理；
- 数据质量是否可靠；
- 模型结果是否稳定；
- 预测概率是否校准；
- 是否存在误用风险；
- 模型局限性是否被充分识别和文档化。

因此，本项目选择银行风险管理中典型的信用风险 PD 模型作为切入点，基于公开数据集完成模型开发、模型验证、压力测试、Expected Loss 计算和验证报告撰写。

本项目不是生产级银行项目，也不使用真实银行内部数据。它的定位是一个可复现的 academic / self-directed project，用于展示对信用风险模型验证流程的理解。

## 2. 项目目标

本项目希望回答以下问题：

1. 如何基于借款人信息构建一个 PD 模型？
2. 如何从模型验证视角评估模型，而不是只看分类准确率？
3. Logistic Regression 和 XGBoost 作为 baseline / challenger model 时，应该如何比较？
4. 为什么信用风险 PD 模型不仅要看 AUC，也要看 calibration？
5. 如何设计简化版压力情景，并分析组合 PD 与 Expected Loss 的变化？
6. 如何形成一份结构化的模型验证报告？

## 3. 数据集说明

### 3.1 数据来源

本项目使用 UCI Machine Learning Repository 的 Statlog German Credit Data。

数据集链接：

- UCI German Credit Data: https://archive.ics.uci.edu/dataset/144/statlog%2Bgerman%2Bcredit%2Bdata

原始数据包含 1,000 条借款人记录，每条记录描述一个信贷申请人的信用状态、贷款期限、贷款金额、储蓄账户、就业状态、住房情况等信息。

### 3.2 目标变量定义

原始 `german.data` 文件里的最后一列叫 `target`，它的含义如下：

- `1`: good credit
- `2`: bad credit

为了把问题转化为“预测违约概率（PD）”，本项目新建了一个二分类建模目标 `bad`：

| 原始 target | 原始含义 | 建模目标 bad |
|---:|---|---:|
| 1 | good credit | 0 |
| 2 | bad credit | 1 |

也就是说，**原始标签 `2` 会被映射为 `bad = 1`，原始标签 `1` 会被映射为 `bad = 0`**。

因此模型输出的概率可以解释为借款人的 PD：

```text
PD = P(bad = 1 | borrower features)
```

### 3.3 原始编码样本如何解读

原始 `german.data` 文件没有表头，每一行由 21 个字段组成。前 20 个字段是借款人特征，第 21 个字段是原始标签 `target`：

```text
target = 1 表示 good credit，在建模时转换为 bad = 0
target = 2 表示 bad credit，在建模时转换为 bad = 1
```

为了让原始编码更容易理解，下面给出 3 条差异较大的样本解析。

#### 示例 1：低金额、短期限、最终为 good credit

原始记录：

```text
A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1
```

逐字段解释：

| 字段 | 原始值 | 含义 |
|---|---|---|
| checking_status | A11 | 支票账户余额 < 0 DM |
| duration_months | 6 | 贷款期限 6 个月 |
| credit_history | A34 | 关键账户 / 其他地方也有信用记录 |
| purpose | A43 | 用途：radio/television |
| credit_amount | 1169 | 贷款金额 1169 |
| savings_status | A65 | 储蓄账户未知 / 无储蓄账户 |
| employment_since | A75 | 当前工作年限 >= 7 年 |
| installment_rate_pct | 4 | 分期还款占可支配收入比例为 4 |
| personal_status_sex | A93 | 男性，单身 |
| other_debtors | A101 | 无共同借款人/担保人 |
| present_residence_since | 4 | 当前住所居住年限为 4 |
| property | A121 | 房地产 |
| age_years | 67 | 年龄 67 岁 |
| other_installment_plans | A143 | 无其他分期付款计划 |
| housing | A152 | 自有住房 |
| existing_credits | 2 | 当前银行已有信用数量为 2 |
| job | A173 | 技术工/正式雇员 |
| num_dependents | 1 | 需赡养人数为 1 |
| telephone | A192 | 有本人名下登记电话 |
| foreign_worker | A201 | 外籍劳工：是 |
| target | 1 | good credit |

翻译成自然语言：

> 一个 67 岁、男性单身、有长期工作、拥有自住房和房地产、贷款 1169、期限 6 个月、用途为收音机/电视、最终被标记为 good credit 的借款人样本。

#### 示例 2：高金额、长期限、年轻借款人，最终为 bad credit

原始记录：

```text
A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2
```

逐字段解释：

| 字段 | 原始值 | 含义 |
|---|---|---|
| checking_status | A12 | 支票账户余额 0 到 200 DM |
| duration_months | 48 | 贷款期限 48 个月 |
| credit_history | A32 | 现有贷款均按期偿还 |
| purpose | A43 | 用途：radio/television |
| credit_amount | 5951 | 贷款金额 5951 |
| savings_status | A61 | 储蓄账户 < 100 DM |
| employment_since | A73 | 当前工作年限 1 到 4 年 |
| installment_rate_pct | 2 | 分期还款占可支配收入比例为 2 |
| personal_status_sex | A92 | 女性，离异/分居/已婚 |
| other_debtors | A101 | 无共同借款人/担保人 |
| present_residence_since | 2 | 当前住所居住年限为 2 |
| property | A121 | 房地产 |
| age_years | 22 | 年龄 22 岁 |
| other_installment_plans | A143 | 无其他分期付款计划 |
| housing | A152 | 自有住房 |
| existing_credits | 1 | 当前银行已有信用数量为 1 |
| job | A173 | 技术工/正式雇员 |
| num_dependents | 1 | 需赡养人数为 1 |
| telephone | A191 | 无本人名下登记电话 |
| foreign_worker | A201 | 外籍劳工：是 |
| target | 2 | bad credit |

翻译成自然语言：

> 一个 22 岁、女性、工作年限 1 到 4 年、贷款 5951、期限 48 个月、用途为收音机/电视、储蓄账户较低、无本人名下电话、最终被标记为 bad credit 的借款人样本。

#### 示例 3：有担保人、租房、存在其他银行分期计划，最终为 bad credit

原始记录：

```text
A11 16 A34 A40 2625 A61 A75 2 A93 A103 4 A122 43 A141 A151 1 A173 1 A192 A201 2
```

逐字段解释：

| 字段 | 原始值 | 含义 |
|---|---|---|
| checking_status | A11 | 支票账户余额 < 0 DM |
| duration_months | 16 | 贷款期限 16 个月 |
| credit_history | A34 | 关键账户 / 其他地方也有信用记录 |
| purpose | A40 | 用途：new car |
| credit_amount | 2625 | 贷款金额 2625 |
| savings_status | A61 | 储蓄账户 < 100 DM |
| employment_since | A75 | 当前工作年限 >= 7 年 |
| installment_rate_pct | 2 | 分期还款占可支配收入比例为 2 |
| personal_status_sex | A93 | 男性，单身 |
| other_debtors | A103 | 有担保人 |
| present_residence_since | 4 | 当前住所居住年限为 4 |
| property | A122 | 建房储蓄协议/人寿保险 |
| age_years | 43 | 年龄 43 岁 |
| other_installment_plans | A141 | 其他分期付款计划：银行 |
| housing | A151 | 租房 |
| existing_credits | 1 | 当前银行已有信用数量为 1 |
| job | A173 | 技术工/正式雇员 |
| num_dependents | 1 | 需赡养人数为 1 |
| telephone | A192 | 有本人名下登记电话 |
| foreign_worker | A201 | 外籍劳工：是 |
| target | 2 | bad credit |

翻译成自然语言：

> 一个 43 岁、男性单身、长期工作但支票账户余额为负、贷款 2625、期限 16 个月、用途为新车、有担保人、租房、还存在其他银行分期付款计划、最终被标记为 bad credit 的借款人样本。

### 3.4 特征类型

数值型变量包括：

- `duration_months`
- `credit_amount`
- `installment_rate_pct`
- `present_residence_since`
- `age_years`
- `existing_credits`
- `num_dependents`

类别型变量包括：

- `checking_status`
- `credit_history`
- `purpose`
- `savings_status`
- `employment_since`
- `personal_status_sex`
- `other_debtors`
- `property`
- `other_installment_plans`
- `housing`
- `job`
- `telephone`
- `foreign_worker`

## 4. 项目结构

```text
credit_risk_pd_validation/
  data/
    raw/
      german.data
  outputs/
    figures/
      target_distribution.png
      credit_amount_distribution.png
      bad_rate_by_duration.png
      roc_curve.png
      calibration_curve.png
      metric_comparison.png
      feature_importance.png
      stress_pd.png
      stress_expected_loss.png
    credit_risk_pd_model_validation_report.pdf
    model_validation_metrics.csv
    stress_testing_results.csv
    feature_importance.csv
    metrics_summary.json
  src/
    run_credit_risk_validation.py
  README.md
  PROJECT_WALKTHROUGH_ZH.md
  interview_notes_zh.md
  requirements.txt
```

## 5. 环境与依赖

项目主要依赖：

```text
pandas
numpy
scikit-learn
matplotlib
xgboost
reportlab
```

安装依赖：

```bash
pip install -r requirements.txt
```

运行项目：

```bash
python src/run_credit_risk_validation.py
```

运行后会自动生成：

- 模型验证指标 CSV；
- 压力测试结果 CSV；
- 特征重要性 CSV；
- 图表；
- PDF 模型验证报告；
- JSON 指标摘要。

## 6. 操作流程总览

整个项目流程如下：

```text
读取数据
  -> 定义 bad = 1 的 PD 建模目标
  -> 数据质量与探索性分析
  -> 构建 Logistic Regression baseline
  -> 构建 XGBoost challenger model
  -> 评估 AUC / KS / Brier / Calibration / CV stability
  -> 基于 cost matrix 选择阈值
  -> 比较 baseline 与 challenger
  -> 分析模型驱动因素
  -> 构造 Base / Adverse / Severe 压力情景
  -> 计算 PD / LGD / EAD / Expected Loss
  -> 输出模型验证报告
```

## 7. 数据探索与质量检查

### 7.1 样本分布

首先检查 good credit 和 bad credit 的比例。

![Target Distribution](outputs/figures/target_distribution.png)

从图中可以看到，数据集中 good credit 占比约 70%，bad credit 占比约 30%。这意味着样本存在一定类别不平衡，但并非极端不平衡。

在信用风险建模中，目标变量分布非常重要。如果坏样本比例过低，模型可能倾向于预测大多数样本为好客户，从而在准确率上看似较高，但实际无法识别风险客户。因此本项目没有使用 accuracy 作为核心指标，而是重点关注：

- AUC；
- KS；
- Brier Score；
- Calibration Curve；
- Cost-sensitive threshold。

### 7.2 贷款金额分布

![Credit Amount Distribution](outputs/figures/credit_amount_distribution.png)

贷款金额是信用风险中非常重要的变量，因为它不仅影响违约风险，也影响违约后的风险暴露 EAD。在本项目中，后续 Expected Loss 计算使用 `credit_amount` 近似 EAD。

### 7.3 贷款期限与坏账率

![Bad Rate by Duration](outputs/figures/bad_rate_by_duration.png)

贷款期限越长，借款人未来状态的不确定性越高，因此在很多信用风险场景中，较长期限可能对应更高风险。本项目通过 observed bad rate by duration 对这一关系进行探索。

## 8. 特征工程与预处理

项目中对数值变量和类别变量分别进行预处理。

### 8.1 数值变量处理

数值变量使用：

```text
SimpleImputer(strategy="median")
StandardScaler()
```

处理逻辑：

- 使用中位数填补缺失值；
- 使用标准化让不同量纲的数值变量处于可比较尺度；
- 对 Logistic Regression 尤其重要，因为线性模型对特征尺度更敏感。

### 8.2 类别变量处理

类别变量使用：

```text
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

处理逻辑：

- 使用众数填补缺失值；
- 使用 One-Hot Encoding 将类别变量转换为模型可处理的数值矩阵；
- `handle_unknown="ignore"` 可以避免测试集或未来数据中出现未知类别时报错。

### 8.3 训练集与测试集划分

项目使用 stratified train/test split：

```text
test_size = 0.30
random_state = 42
stratify = y
```

使用 stratify 的原因是保持训练集和测试集中 bad credit 比例一致，避免由于随机划分导致测试集风险分布偏移。

## 9. 模型设计

本项目构建两个模型：

1. Logistic Regression baseline
2. XGBoost challenger model

### 9.1 Logistic Regression Baseline

Logistic Regression 在信用风险建模中具有较强解释性，常被用于传统信用评分场景。

优点：

- 模型结构简单；
- 输出概率自然可解释为 PD；
- 系数方向可以辅助解释变量影响；
- 更容易进行模型治理和验证。

在模型验证语境下，baseline model 的价值不是追求复杂度，而是提供一个清晰、可解释、可挑战的参考模型。

### 9.2 XGBoost Challenger Model

XGBoost 用作 challenger model，用于检验非线性关系和变量交互是否能提升预测表现。

优点：

- 能捕捉非线性关系；
- 能处理复杂变量交互；
- 通常具有较强预测能力。

但在模型风险管理中，复杂模型也带来额外挑战：

- 可解释性更弱；
- 校准可能不稳定；
- 治理成本更高；
- 需要更多监控和文档支持。

因此，本项目不会简单地认为“复杂模型一定更好”，而是从 AUC、KS、Brier、校准、稳定性和可解释性多个角度比较两个模型。

## 10. 模型验证指标

### 10.1 AUC

AUC 衡量模型对好坏客户的排序能力。AUC 越高，说明模型越能把高风险客户排在低风险客户前面。

但 AUC 只反映排序能力，不反映概率是否准确。例如，一个模型可以排序很好，但预测 PD 系统性偏高或偏低。

### 10.2 KS

KS 衡量 good / bad 两类样本累积分布的最大差异，是信用评分模型中常用的区分度指标。

KS 越高，说明模型对好坏客户的分离能力越强。

### 10.3 Brier Score

Brier Score 衡量预测概率与真实结果之间的均方误差。

在 PD 模型中，Brier Score 非常重要，因为 PD 是概率，不只是分类标签。若模型用于 Expected Loss 或风险定价，概率校准质量会直接影响风险估计。

### 10.4 Calibration Curve

Calibration Curve 比较 predicted PD 和 observed bad rate。

如果模型校准良好，那么预测 PD 为 20% 的样本组，其实际坏账率也应接近 20%。

### 10.5 Cross-validation AUC Stability

交叉验证用于观察模型在不同训练/验证划分下的 AUC 稳定性。

如果 CV AUC 标准差过大，说明模型对样本划分敏感，稳定性不足。

### 10.6 Cost-sensitive Threshold

German Credit 数据集附带一个重要设定：将坏客户误判为好客户的成本高于将好客户误判为坏客户。

在本项目中：

```text
False Negative: bad customer predicted as good, cost = 5
False Positive: good customer predicted as bad, cost = 1
```

因此，项目没有默认使用 0.5 作为分类阈值，而是寻找使误判成本最低的 threshold。

## 11. 模型验证结果

模型主要结果如下：

| Model | AUC | KS | Brier | CV AUC Mean | CV AUC Std | Cost Threshold | Expected Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8011 | 0.4937 | 0.1574 | 0.7640 | 0.0241 | 0.245 | 0.4967 |
| XGBoost Challenger | 0.7834 | 0.4317 | 0.1659 | 0.7713 | 0.0354 | 0.140 | 0.5100 |

### 11.1 ROC 曲线

![ROC Curve](outputs/figures/roc_curve.png)

Logistic Regression 的 AUC 约为 0.801，XGBoost 的 AUC 约为 0.783。从测试集表现看，Logistic Regression 在区分度上略优。

这也说明，在小样本、结构化信用数据上，复杂模型不一定优于简单模型。

### 11.2 Calibration Curve

![Calibration Curve](outputs/figures/calibration_curve.png)

Calibration Curve 用于观察 predicted PD 与 observed bad rate 的一致性。

本项目中，Logistic Regression 的 Brier Score 为 0.1574，XGBoost 为 0.1659。Brier Score 越低越好，因此 Logistic Regression 在概率校准方面略优。

对于 PD 模型来说，这一点非常重要。因为如果模型只是用于排序，AUC 可能足够；但如果模型输出要用于 Expected Loss 计算，那么 PD 的概率质量必须被验证。

### 11.3 指标对比

![Metric Comparison](outputs/figures/metric_comparison.png)

综合 AUC、KS、Brier、成本阈值和可解释性，本项目最终选择 Logistic Regression 作为 preferred model。

选择理由：

1. AUC 更高；
2. KS 更高；
3. Brier Score 更低；
4. 可解释性更强；
5. 更适合模型验证报告中的 conceptual soundness 和 governance discussion。

这并不意味着 XGBoost 不好，而是说明 challenger model 没有在综合验证指标上明显超过 baseline。

## 12. 模型驱动因素分析

![Feature Importance](outputs/figures/feature_importance.png)

模型重要变量包括：

- `purpose_A46`
- `checking_status_A14`
- `credit_history_A34`
- `property_A124`
- `savings_status_A61`
- `employment_since_A74`
- `foreign_worker_A202`
- `checking_status_A11`

在真实模型验证中，特征重要性分析只是第一步，还需要进一步检查：

- 变量方向是否符合业务逻辑；
- 是否存在代理变量或合规风险；
- 是否存在稳定性问题；
- 是否需要 reason code；
- 是否存在公平性或歧视性风险。

本项目作为 academic project，没有展开公平性测试，但在验证报告中将其列为后续改进方向。

## 13. 压力测试设计

### 13.1 压力测试动机

信用风险模型不应只在正常样本分布下被评估，也需要观察在不利情景下组合风险如何变化。

本项目构造三个情景：

| Scenario | Description |
|---|---|
| Base | 原始组合 |
| Adverse | 部分高风险客户贷款金额、期限和风险状态恶化 |
| Severe | 更大比例、更强幅度的风险恶化 |

### 13.2 压力变量设计

压力情景主要调整：

- `credit_amount`
- `duration_months`
- `checking_status`
- `savings_status`
- `employment_since`
- `credit_history`
- `property`

设计逻辑是：当宏观或借款人状态恶化时，贷款期限、贷款金额和信用状态可能同时变差，从而推高组合 PD。

需要强调：这不是正式 CCAR 压力测试模型，而是 educational scenario analysis。真实监管压力测试需要宏观变量、情景路径、治理流程和更严格的模型验证。

## 14. Basel 风格 Expected Loss 映射

本项目使用简化的 Basel 风格风险参数框架：

```text
Expected Loss = PD x LGD x EAD
```

其中：

- PD: 模型预测的违约概率；
- LGD: 假设为 45%；
- EAD: 使用 credit amount 近似。

在真实银行场景中，LGD 和 EAD 通常需要单独建模或验证，本项目为了聚焦 PD 模型验证，对 LGD 和 EAD 做了简化处理。

## 15. 压力测试结果

| Scenario | Mean PD | Portfolio EAD | Expected Loss | PD Uplift vs Base | EL Uplift vs Base |
|---|---:|---:|---:|---:|---:|
| Base | 0.3008 | 3,271,258 | 514,878 | 0.0000 | 0.0000 |
| Adverse | 0.4156 | 3,440,094 | 921,262 | 0.3818 | 0.7893 |
| Severe | 0.5491 | 3,896,741 | 1,298,799 | 0.8257 | 1.5225 |

### 15.1 组合 PD 压力测试

![Stress PD](outputs/figures/stress_pd.png)

Base 情景下平均 PD 约为 30.1%，与样本坏账率 30.0% 接近，说明 Logistic Regression 的整体概率水平较合理。

在 Adverse 情景下，平均 PD 上升至约 41.6%；在 Severe 情景下，平均 PD 上升至约 54.9%。

### 15.2 Expected Loss 压力测试

![Stress Expected Loss](outputs/figures/stress_expected_loss.png)

Expected Loss 在压力情景下显著上升。Severe 情景下 EL uplift 超过 150%。

这说明：

- PD 上升会推高风险损失；
- EAD 上升会进一步放大损失；
- 组合风险对 borrower quality 和 exposure 同时敏感。

## 16. 模型验证报告

项目自动生成 PDF 验证报告：

```text
outputs/credit_risk_pd_model_validation_report.pdf
```

报告结构包括：

1. Executive Summary
2. Model Purpose and Intended Use
3. Data Description and Data Quality Review
4. Methodology Review
5. Model Performance Testing
6. Calibration and Discrimination Analysis
7. Benchmarking
8. Stress Testing and Sensitivity Analysis
9. Model Limitations
10. Validation Findings and Recommendations

该结构参考了模型风险管理中的常见验证逻辑，尤其是：

- conceptual soundness；
- outcomes analysis；
- benchmarking；
- model limitations；
- validation documentation。

参考资料：

- Federal Reserve SR 11-7 Model Risk Management Guidance: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- Basel II Comprehensive Version, BIS: https://www.bis.org/publ/bcbs128.pdf

## 17. 主要验证发现

### Finding 1: Logistic Regression 综合表现优于 XGBoost challenger

虽然 XGBoost 是更复杂的非线性模型，但在本项目测试集上，Logistic Regression 的 AUC、KS 和 Brier Score 表现更好。

这说明在小样本信用风险数据中，复杂模型不一定带来更优的验证结果。

### Finding 2: PD 模型必须关注校准

AUC 只能说明排序能力，不能说明概率是否准确。

对于信用风险 PD 模型，如果输出概率要用于 Expected Loss、风险定价或资本计量，就必须检查 calibration。

### Finding 3: 成本敏感阈值比默认 0.5 更合理

在信用风险中，错把坏客户判断为好客户通常比错拒好客户成本更高。因此本项目使用 cost matrix 寻找成本最小化阈值。

### Finding 4: 压力情景下组合风险显著上升

在 Adverse 和 Severe 情景下，组合平均 PD 和 Expected Loss 均显著上升，说明模型能够反映压力情景下的风险变化。

### Finding 5: 项目存在明确局限性

本项目使用公开小样本数据集，没有真实银行业务数据、时间序列 out-of-time validation、宏观变量或生产级监控数据。因此它不能用于真实授信决策。

## 18. 项目局限性

本项目的主要局限包括：

1. 数据集规模较小，仅 1,000 条样本；
2. 没有时间维度，无法做真正的 out-of-time validation；
3. 缺少宏观经济变量，压力测试为规则化情景设计；
4. LGD 固定为 45%，没有单独建模；
5. EAD 使用 credit amount 近似，较为简化；
6. 没有进行 fairness、bias、reject inference 等更深入验证；
7. 不是生产级银行模型，不能直接用于真实风险决策。

这些局限性并不削弱项目价值，反而体现模型验证中的一个重要原则：模型使用边界必须被清楚识别和记录。

## 19. 面试讲解口径

可以用下面这段话介绍项目：

```text
这个项目是我研究生阶段做量化建模和模型评估时的一个拓展项目，不是企业生产项目。
我之前做过数学建模竞赛、机器学习模型训练和模型评估，因此进一步选择信用风险 PD 模型这个金融风险典型场景，系统练习模型验证流程。
项目中我使用 Logistic Regression 作为可解释 baseline，XGBoost 作为 challenger model，从数据质量、AUC、KS、校准、交叉验证、误判成本、压力测试和模型局限性角度进行验证，并输出模型验证报告。
我希望通过这个项目证明，我不仅会训练模型，也理解模型风险管理中 effective challenge、quantitative testing 和 documentation 的要求。
```

## 20. 简历写法

中文版本：

```text
研究生阶段量化建模拓展项目：信用风险 PD 模型验证与压力测试
- 在研究生阶段数学建模与机器学习模型评估经历基础上，选取金融风险中典型的信用风险 PD 模型作为拓展方向，基于 UCI German Credit Data 构建并验证借款人违约概率模型。
- 使用 Logistic Regression 作为可解释 baseline，XGBoost 作为 challenger model，从数据质量、变量合理性、模型区分度、校准效果、交叉验证稳定性和误判成本角度开展模型验证。
- 使用 AUC、KS、Brier Score、Calibration Curve、Confusion Matrix 等指标评估模型表现，并比较模型在预测能力、概率校准和可解释性方面的差异。
- 设计 Base/Adverse/Severe 压力情景，分析组合 PD 与 Expected Loss 的变化，并基于 PD/LGD/EAD 框架计算预期损失。
- 输出模型验证报告，覆盖方法论、测试结果、benchmark、模型局限性和改进建议。
```

英文版本：

```text
Graduate Quantitative Modeling Extension Project: Credit Risk PD Model Validation and Stress Testing
- Built and independently validated borrower-level probability of default models using UCI German Credit Data, extending prior graduate-level modeling and machine learning evaluation experience into a credit risk model validation setting.
- Used Logistic Regression as an interpretable baseline and XGBoost as a challenger model; evaluated data quality, variable reasonableness, discrimination, calibration, cross-validation stability, and cost-sensitive thresholds.
- Assessed model performance using AUC, KS, Brier Score, calibration curves, confusion matrix analysis, and benchmarking between baseline and challenger models.
- Designed base/adverse/severe stress scenarios and mapped PD to expected loss using Basel-style PD/LGD/EAD concepts.
- Produced a validation report covering methodology, test results, benchmarking, model limitations, and recommendations.
```

## 21. 如何复现

### 21.1 安装依赖

```bash
pip install -r requirements.txt
```

### 21.2 运行脚本

```bash
python src/run_credit_risk_validation.py
```

### 21.3 查看输出

```text
outputs/
  credit_risk_pd_model_validation_report.pdf
  model_validation_metrics.csv
  stress_testing_results.csv
  feature_importance.csv
  metrics_summary.json
  figures/
```

## 22. 总结

本项目的核心价值不是证明模型有多复杂，而是展示一套完整的模型验证思维：

- 明确模型用途；
- 检查数据质量；
- 构建 baseline 和 challenger；
- 使用多维指标验证模型；
- 关注概率校准；
- 进行压力测试；
- 识别模型局限；
- 输出验证报告。

这与 Quantitative Risk / Model Validation 岗位中的工作要求高度相关，尤其是信用风险模型、定量测试、压力测试、模型文档和模型局限性分析。
