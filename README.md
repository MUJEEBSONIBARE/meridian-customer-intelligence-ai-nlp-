# Customer Intelligence with AI, Machine Learning and NLP: The Explainability Challenge

## Overview

This project develops an AI-powered customer-intelligence workflow for a UK financial-services case study. It combines structured-data analysis, supervised machine learning, Azure AI Language, SHAP explainability, fairness auditing, and Power BI reporting to investigate complaint-escalation risk and customer feedback.

The objective is not simply to predict which customers may escalate a complaint. The project also examines how model outputs can be explained, how performance differs across customer groups, and how analytical insights can be presented to customer-operations and compliance stakeholders.

The work is an analytical prototype. Model outputs should support trained staff in prioritising review and potential proactive support; they should not trigger automated adverse decisions about customers.

## Business Problem

Customer complaints can create operational costs, harm customer relationships, and require additional investigation. Customer-operations teams need ways to understand escalation patterns, identify cases that may warrant review, and detect recurring themes in support interactions.

This project addresses three connected questions:

1. **Escalation risk:** Can structured customer and outcome data provide a useful signal for identifying customers who may escalate a complaint?
2. **Customer voice:** What do support-ticket texts and customer reviews reveal about sentiment and recurring issues?
3. **Responsible use:** How can the model be explained, assessed for differences in performance across groups, and presented as human-reviewed decision support?

## Project Objectives

- Profile and validate structured customer and complaint-outcome data.
- Explore escalation patterns across products, age bands, and complaint history.
- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate model discrimination and classification behaviour using metrics relevant to binary classification.
- Compare operating thresholds using estimated business costs for false negatives and false positives.
- Apply Azure AI Language to support-ticket sentiment analysis and key-phrase extraction.
- Use SHAP to examine global feature importance and individual predictions.
- Audit recall and false-positive rates across age bands and product types.
- Integrate analytical outputs into a four-page Power BI dashboard for business and responsible-AI monitoring.

## Data and Scope

The analysis uses four datasets representing customer profiles, complaint outcomes, support tickets, and customer reviews. The structured customer and outcome data are joined using `customer_id`. The text datasets support sentiment and key-phrase analysis.

The analysis covers:

- **500 customer profiles** and **500 outcome records**, merged into a dataset with 500 customers and 12 fields.
- **919 support tickets** analysed using Azure AI Language.
- **310 customer reviews** used in the customer-feedback analysis.



## Methodology

### 1. Data Preparation and Exploratory Analysis

The customer and outcome datasets were merged on `customer_id` using an inner join with one-to-one validation. The merged data was checked for missing values and duplicate rows before modelling.

Exploratory analysis examined escalation rates by product type and age band, as well as the relationship between previous complaints and escalation. This established the business context before fitting classification models.

**Key findings**

- Overall escalation rate: **26.8%**.
- Personal Loans had the highest reported product-level escalation rate at **36.5%**.
- Home Insurance followed at **31.0%**, with Credit Cards at **29.6%**.
- Mortgages had the lowest reported product-level escalation rate at **18.9%**.
- **60.0%** of customers with four previous complaints escalated, compared with **25.6%** of customers with no previous complaints.

These are descriptive associations. They should not be interpreted as proof that a particular feature causes escalation.

### 2. Complaint-Escalation Classification

The target variable was `escalated`. The reported feature set included:

- `tenure_months`
- `account_value_gbp`
- `previous_complaints`
- `age_band`
- `product_type`
- `primary_channel`
- `region`

Categorical variables were one-hot encoded. The data was split into training and test sets using an 80/20 stratified split, and numeric features were standardised. Logistic Regression and Random Forest were trained with balanced class weights.

#### Model Comparison

| Model | Test AUC | Interpretation |
|---|---:|---|
| Logistic Regression | 0.486 | Weak discrimination |
| Random Forest | 0.525 | Higher AUC than Logistic Regression, but still weak discrimination |

Random Forest was selected for further operating-threshold analysis because it achieved the higher AUC of the two models. Its performance was nevertheless too weak to support unrestricted operational use.

#### Threshold and Business-Cost Analysis

The project considered two types of classification error:

- **False negative:** A customer who escalates is not flagged. The analysis assigned an estimated cost of **£800** per case.
- **False positive:** A customer is flagged but does not escalate. The analysis assigned an estimated cost of **£25** in staff time.

| Threshold | True Positives | False Positives | False Negatives | True Negatives | Recall | Precision | Estimated Cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 21 | 55 | 6 | 18 | 77.8% | 27.6% | £30,875 |
| 0.40 | 16 | 37 | 11 | 36 | 59.3% | 30.2% | £48,625 |
| 0.50 | 7 | 19 | 20 | 54 | 25.9% | 26.9% | £82,375 |

A threshold of **0.30** had the lowest estimated cost among the thresholds tested and the highest recall. However, its precision of 27.6% means that most customers flagged at this threshold did not escalate in the reported test sample.

Consequently, a model flag should be treated as a prompt for human review, not as proof that a customer will escalate. The costs are estimates based on the assumptions used in the analysis, not measured realised financial savings.

### 3. NLP: Sentiment and Key-Phrase Extraction

Azure AI Language was used to analyse **919 support tickets**, providing sentiment classifications and extracted key phrases.

#### Ticket Sentiment

| Sentiment Classification | Number of Tickets |
|---|---:|
| Negative | 628 |
| Neutral | 269 |
| Positive | 17 |
| Mixed | 5 |
| **Total** | **919** |

The output was predominantly negative, indicating substantial customer dissatisfaction within the analysed ticket corpus.

Service Failure had the highest reported average negative-sentiment confidence in the category analysis, while Product Query was the largest category by volume, with **191 tickets**. These categories represent different operational priorities: the severity of negative feedback and the frequency of an issue should both be considered.

The average sentiment score was reported as **0.690** for non-escalated tickets and **0.731** for escalated tickets, a difference of **0.041**. This difference is directionally consistent with more negative sentiment among escalated cases, but it is not sufficient to use sentiment as a standalone escalation predictor.

#### Key Phrases

| Phrase | Occurrences |
|---|---:|
| account | 23 |
| home insurance policy | 9 |
| excess | 9 |
| billing dispute | 7 |
| monthly fee | 6 |
| notification | 6 |

The phrases highlight recurring account, insurance, billing, and communication themes in the analysed ticket text. Key-phrase counts indicate frequency, not necessarily severity or root cause; they should be interpreted alongside ticket context and other operational data.

The customer review-star trend fluctuated over the 13-month period from May 2025 to May 2026, without a consistent improvement or deterioration in the reported analysis.

### 4. Explainability with SHAP

SHAP was used to examine model behaviour at both global and individual levels.

The global feature-importance analysis identified `tenure_months`, `previous_complaints`, and `account_value_gbp` among the strongest contributors, alongside product, region, and channel features.

Individual explanations were used to examine different types of prediction:

- **True positive — CUS-0350:** The model correctly flagged the customer at a predicted probability of 57%. Previous complaints increased the score by approximately 0.07, while tenure and the 46–55 age-band feature reduced it by approximately 0.05 and 0.04.
- **False positive — CUS-0207:** The model flagged the customer at 50%, although the customer did not escalate. This example illustrates how feature contributions can combine to produce an incorrect flag.
- **False negative — CUS-0136:** The customer escalated but received a predicted probability of 18%. The explanation showed several features contributing to a lower score, demonstrating how the model can miss an escalation.

These explanations help reviewers understand which model features influenced a prediction. They do not establish causation, prove that a prediction is correct, or justify a customer outcome by themselves.

### 5. Fairness Audit

The audit compared recall and false-positive rates across age bands and product types. Age band was treated as a fairness-sensitive attribute and included in the audit to identify potential differences in model performance.

#### Age-Band Results

Reported recall ranged from **0%** for the 56–65 group to **100%** for the 65+ group. False-positive rates ranged from **40.0%** for the 65+ group to **94.4%** for the 46–55 group.

#### Product-Level Results

Reported recall ranged from **0%** for Mortgages to **100%** for Personal Loans and Savings Accounts.

These differences are material, but some subgroups have small test-sample sizes. The estimates should therefore be treated as indicative rather than conclusive. The variation nevertheless reinforces the need for additional validation and ongoing monitoring before considering any production use.

The audit does not establish that the model is fair. It identifies observed disparities that require investigation.

### 6. Power BI Customer-Intelligence Dashboard

A four-page Power BI report brought together the model outputs, NLP findings, and responsible-AI monitoring.

#### Page 1 — Escalation Risk Overview

Provides a high-risk customer KPI, product and risk-band slicers, a risk-band-by-product visual, a customer table supporting drill-through, and a Decomposition Tree for interactive exploration.

A geographic map was attempted, but the region labels could not be plotted reliably by the map service. The report documents this limitation rather than presenting a potentially misleading map. A “flagged this month” KPI was not calculated because the source data did not contain a flag-date field.

#### Page 2 — Customer Detail

Displays customer-level information, including risk score, previous complaints, latest ticket sentiment, and tenure. Customer ID supports drill-through, with back navigation to the overview.

#### Page 3 — Sentiment & NLP Themes

Shows monthly sentiment trends, sentiment by ticket category, extracted customer key phrases, and customer star-rating trends.

#### Page 4 — Responsible AI Monitoring

Presents escalation rate versus model recall by age band, false-positive rates by group, global SHAP feature importance, and a compliance statement describing the model's limitations and intended use.

Row-level security was configured and tested using a regional compliance role. The machine-learning and NLP outputs were imported into Power BI and related through `customer_id`.

## Key Findings

1. **Predictive performance is weak.** Random Forest outperformed Logistic Regression on AUC, but its AUC of 0.525 indicates limited discrimination.
2. **Threshold choice changes the operational trade-off.** The 0.30 threshold reduced the estimated cost in the tested comparison and improved recall, but produced many false positives.
3. **Customer feedback provides useful context.** Ticket sentiment and key phrases reveal recurring service and product themes, but sentiment alone is not a strong escalation signal.
4. **Model performance varies across groups.** Differences in recall and false-positive rates require further investigation and careful monitoring.
5. **Explainability supports scrutiny, not certainty.** SHAP helps explain model outputs but does not prove causation or eliminate model risk.
6. **The dashboard brings technical outputs into a business-facing view.** Power BI connects escalation-risk analysis, customer-level exploration, NLP trends, and responsible-AI monitoring.

## Responsible AI and Intended Use

The model is intended to help trained customer-operations staff identify cases that may warrant further review and potential proactive support. It is not a definitive prediction of an individual customer's future behaviour.

The model must not be used as an automated decision-maker for customer eligibility, pricing, access to services, complaint outcomes, or other adverse customer decisions. Staff should review the wider customer context, consider operational evidence, and be able to override model flags. No customer outcome should be triggered solely by the model.

Age band was treated as a fairness-sensitive attribute and explicitly audited. Other available features may also act as proxies or be associated with unequal outcomes, so monitoring should continue across relevant groups.

## Limitations

- The reported AUC scores show weak discrimination.
- The test sample and subgroup sizes are limited, making some fairness estimates unstable.
- The 0.30 threshold has low precision and could create a substantial review burden.
- The estimated business costs depend on assumed costs for false negatives and false positives.
- Observational patterns and SHAP contributions do not establish causation.
- Sentiment classifications and extracted phrases can lose context, ambiguity, or nuance in customer language.
- The geographic map and a flag-date KPI could not be reliably produced from the available source fields.
- The case-study datasets are not verified real customer records, so the findings should not be generalised to a real institution without independent validation.

## Recommendations and Next Steps

1. **Validate on a larger independent dataset.** Reassess discrimination, calibration, threshold performance, and the stability of subgroup metrics.
2. **Investigate alternative modelling approaches.** Compare additional baselines and feature/threshold strategies without compromising interpretability or responsible use.
3. **Review fairness disparities.** Investigate groups with particularly low recall or high false-positive rates and document appropriate mitigation options.
4. **Keep human oversight mandatory.** Treat model outputs as review prompts, not as automatic decisions or definitive customer labels.
5. **Establish ongoing monitoring.** Track predictive performance, data quality, review outcomes, and subgroup differences over time.
6. **Improve operational data capture.** Add reliable flag timestamps if a “flagged this month” metric is required, and use validated geographic fields if geographic analysis is needed.
7. **Use NLP as a supporting signal.** Combine sentiment and key-phrase trends with ticket context and operational evidence rather than using them as standalone decision rules.

## Conclusion

This project demonstrates an end-to-end customer-intelligence workflow that combines structured-data analysis, machine learning, Azure AI Language, SHAP explainability, fairness auditing, and Power BI reporting.

Its most important result is not that the model is ready for deployment, but that predictive performance, subgroup differences, and operational error costs must be evaluated before model outputs are used in customer operations. The appropriate position for this prototype is human-reviewed decision support, with stronger independent validation and continued fairness monitoring required before any wider use.
