# 信用风险 PD 模型验证项目：面试准备稿

## 30 秒项目介绍

我做这个项目的目的不是单纯训练一个分类模型，而是模拟模型验证分析师的工作流。我基于 UCI German Credit Data 构建 borrower-level PD 模型，使用 Logistic Regression 作为可解释 baseline，XGBoost 作为 challenger model。随后从数据质量、模型方法、区分度、校准度、稳定性、误判成本、benchmark、压力测试和模型局限性几个角度进行 independent validation，并输出一份模型验证报告。

## 为什么选择这个项目

这个岗位要求 credit risk / market risk 相关的 quantitative analytics 项目经验，也强调 model validation、statistical testing、stress testing、Basel、CCAR 等知识。信用风险 PD 模型是银行风险管理里非常典型的模型类型，所以我选择做一个完整的 PD model validation 项目，来证明我不仅会建模，也理解模型验证的工作方式。

## 项目方法

- 数据集：UCI Statlog German Credit Data，共 1,000 条借款人样本，目标变量为 good/bad credit。
- 目标定义：将 bad credit 映射为 `bad = 1`，模型输出解释为 PD，即 Probability of Default。
- Baseline model：Logistic Regression，优点是可解释性强，适合作为传统信用评分参考模型。
- Challenger model：XGBoost，用于检验非线性关系和变量交互是否提升模型表现。
- 验证指标：AUC、KS、Brier Score、Calibration Curve、Cross-validation AUC、Confusion Matrix、Cost-sensitive threshold。
- 压力测试：构建 Base、Adverse、Severe 三个情景，观察组合平均 PD 和 Expected Loss 的变化。
- Expected Loss 映射：使用 `EL = PD x LGD x EAD`，其中 LGD 假设为 45%，EAD 使用 credit amount 近似。

## 关键结果

- 样本坏账率：30.0%。
- Logistic Regression AUC：约 0.801，KS：约 0.494，Brier Score：约 0.157。
- XGBoost AUC：约 0.783，KS：约 0.432，Brier Score：约 0.166。
- 最终选择 Logistic Regression 作为 preferred model，不是因为它一定最复杂，而是因为它在 AUC、KS、校准和可解释性之间更均衡。
- 压力测试中，Base 平均 PD 约 30.1%，Adverse 约 41.6%，Severe 约 54.9%；Expected Loss 在 Severe 情景下相比 Base 明显上升。

## 可以主动强调的点

1. **模型验证不是只看 AUC。**
   我同时看了校准度、稳定性、误判成本、可解释性和模型局限性。对于 PD 模型来说，概率本身要能解释和使用，所以 Brier Score 和 Calibration Curve 很重要。

2. **Logistic Regression 不一定“落后”。**
   在信用风险场景里，可解释性、稳定性和治理成本非常重要。复杂模型如果只带来有限性能提升，却降低可解释性，未必适合生产使用。

3. **压力测试是模型验证的自然延伸。**
   我设计 adverse/severe 情景，让高风险借款人的期限、金额和风险状态恶化，然后观察组合 PD 和 Expected Loss 的上升。这和监管压力测试的思想一致，但我会明确说这是 educational scenario analysis，不是正式 CCAR 模型。

4. **项目有模型局限性说明。**
   数据集较小、没有时间维度、没有宏观变量，LGD/EAD 是简化假设，所以不能直接用于真实银行决策。这个说明反而体现了模型风险意识。

## 简历中文写法

信用风险 PD 模型验证与压力测试项目：
基于 UCI German Credit Data 构建借款人违约概率（PD）模型，使用 Logistic Regression 作为可解释 baseline，XGBoost 作为 challenger model；从数据质量、变量合理性、模型区分度、校准效果、交叉验证稳定性和误判成本角度开展模型验证，使用 AUC、KS、Brier Score、Calibration Curve、Confusion Matrix 等指标评估模型表现；设计 Base/Adverse/Severe 压力情景，分析组合 PD 与 Expected Loss 的变化，并基于 Basel 风格的 PD/LGD/EAD 框架计算预期损失；输出模型验证报告，覆盖方法论、测试结果、benchmark、模型局限性和改进建议。

## 简历英文写法

Credit Risk PD Model Validation and Stress Testing:
Built and independently validated borrower-level probability of default models using UCI German Credit Data, with Logistic Regression as an interpretable baseline and XGBoost as a challenger model. Evaluated data quality, discrimination, calibration, stability, cost-sensitive thresholds, and model limitations using AUC, KS, Brier Score, calibration curves, cross-validation, and confusion matrix analysis. Designed base/adverse/severe stress scenarios and mapped PD to expected loss using Basel-style PD/LGD/EAD concepts; produced a validation report aligned with SR 11-7-style model risk management principles.

## 可能被问到的问题

**Q: 为什么最后选择 Logistic Regression，而不是 XGBoost？**

A: 因为在这个项目里 Logistic Regression 的 AUC 和 KS 更高，Brier Score 也更好，同时可解释性更强。信用风险模型不仅追求预测效果，也需要稳定、可解释、可治理。XGBoost 可以作为 challenger model，但没有明显证明其综合表现优于 baseline。

**Q: 你怎么理解 PD、LGD、EAD？**

A: PD 是违约概率，LGD 是违约后的损失率，EAD 是违约暴露。Expected Loss 可以用 `EL = PD x LGD x EAD` 估算。这个项目重点是 PD 模型验证，LGD 和 EAD 做了简化假设。

**Q: 这个项目和模型验证岗位有什么关系？**

A: 我模拟了模型验证的完整链路：先明确模型用途和数据口径，再评估模型方法是否合理，然后做 outcome testing，包括 AUC、KS、校准、稳定性和 benchmark，最后做压力测试和局限性说明，并形成验证报告。这和模型验证岗位要求的 effective challenge、quantitative testing 和 documentation 是一致的。

**Q: 这个项目有什么不足？**

A: 数据集规模较小，没有真实银行的时间序列表现和宏观经济变量，压力情景是规则化设计，不是监管级宏观情景；LGD 和 EAD 也没有单独建模。因此它适合作为学习和展示项目，不应被解释为生产级信用风险模型。

