# Instamart Customer Intelligence & Support Copilot

## Overview

This is a reproducible, offline capstone built with Python, pandas, SQLite and scikit-learn. It uses fictional synthetic data only. The brand and policies are a teaching scenario, not real Instamart customer data or official policies. All currency values are INR. No API key, paid model, external network call or hosted AI service is required: every simulated language-model call uses the supplied `mock_llm.py`.

The repository contains four connected parts:

- **Part A — Data Foundation:** audit and clean raw CSV files, create a SQLite database, run SQL analysis and check the expected totals.
- **Part B — Customer Intelligence:** calculate RFM metrics, segment customers, and train/evaluate a lapse model.
- **Part C — Support Triage:** redact personal data, classify tickets with retry/output guardrails, respect a token budget, and route by urgency and customer segment.
- **Part D — Policy RAG and Refund Agent:** retrieve policy chunks, answer grounded questions, and run a resumable tool-using refund agent.

## Reproduce

Use Python 3.11 or newer. Run the commands from the repository root. The commands below are shown for Windows PowerShell; on macOS/Linux activate with `source .venv/bin/activate` instead of the second command.

1. Create an isolated environment:
   ```bash
   python -m venv .venv
   ```
2. Activate it (PowerShell):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install the pinned open-source libraries:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Generate the deterministic synthetic input data. The final line should be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`:
   ```bash
   python generate_data.py
   ```
5. Clean the data and create `data/instamart.db`:
   ```bash
   python src/part_a.py
   ```
6. Run the data validation assertions. Successful output ends with `ALL CHECKPOINTS PASSED`:
   ```bash
   python src/validate.py
   ```
7. Run all seven SQL queries and pandas cross-checks:
   ```bash
   python src/part_a_analysis.py
   ```
8. Build segments and train/evaluate the lapse model:
   ```bash
   python src/part_b.py
   ```
9. Run ticket triage, both prompts and the token-budget demonstration:
   ```bash
   python src/part_c.py
   ```
10. Run policy retrieval/evaluation and the refund-agent scenarios:
    ```bash
    python src/part_d.py
    ```

You can also run `python run_all.py` after activating the environment to execute the complete pipeline in order. `generate_data.py` and `mock_llm.py` are the supplied scripts and are kept as source files in the repository. Outputs and synthetic input files are reproducible; Part C prints retry logs because the mock deliberately injects temporary failures.

## Part A

### Audit and cleaning rules

Raw audit shape, data types, missing counts and percentages are saved in `outputs/raw_audit.json`. The planted raw-data checks are reproduced:

| Raw orders/customer check | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| `pending` rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

Customer cleaning rules: trim city whitespace and title-case city names; map gender variants to `male`, `female`, or `unknown`; treat ages outside 10–100 as missing and fill missing ages with the median of that city using `groupby(...).transform`; parse signup dates. After cleaning there are 1,200 customers, six cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184), genders female 610, male 560, unknown 30, and no missing ages (all filled values are between 18 and 60).

Order cleaning rules, in order: parse `order_ts`; remove exact duplicate rows; sort by timestamp and keep the latest record for each `order_id` so an earlier pending row cannot replace a final status; convert amounts to numeric and reconstruct missing amounts from product price × quantity × (1 − discount percentage); convert delivery time to numeric and turn negative values into missing; fill missing delivery time only for completed orders using the completed-order median for that product category; leave cancelled delivery time missing. The final order table has 12,309 rows: 11,110 completed and 1,199 cancelled, no pending rows, no missing amounts, no missing delivery times for completed orders, and 1,199 missing delivery times for cancelled orders. Completed-order revenue is **INR 2,998,668.80**.

`src/validate.py` asserts 12 checkpoints and prints `ALL CHECKPOINTS PASSED`. Clean CSVs are in `data/clean/`; SQLite tables `customers`, `products`, and `orders` are in `data/instamart.db`.

### SQL and pandas results

Seven runnable queries are in `sql/queries.sql`. Each is executed through `pandas.read_sql_query` and summarized in `outputs/part_a_analysis.json`.

| Query | Result |
|---|---|
| Q1 — WHERE then HAVING | 30 customers |
| Q2 — revenue by city/category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 — products above average catalogue price | 11 products |
| Q4 — customers without completed orders | 25 customers |
| Q5 — customers above average completed spend | 550 customers; average customer spend INR 2,552.06 |
| Q6 — RANK, DENSE_RANK and ROW_NUMBER | 1,175 customers; 8 rows have RANK different from ROW_NUMBER |
| Q7 — daily running total | 183 days; final cumulative revenue INR 2,998,668.80 |

Ties can share the same `RANK()` value while `ROW_NUMBER()` assigns a unique position to each row, so later row numbers can differ from ranks. The top customer per city is: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

The pandas pivot grand total also equals INR 2,998,668.80. City totals, mean order amounts and counts: Bengaluru INR 523,643.40 / INR 276.47 / 1,894; Chennai INR 463,629.15 / INR 272.72 / 1,700; Delhi INR 502,829.80 / INR 243.62 / 2,064; Hyderabad INR 528,589.90 / INR 284.95 / 1,855; Mumbai INR 479,058.50 / INR 297.00 / 1,613; Pune INR 500,918.05 / INR 252.48 / 1,984.

Completed order amounts have mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80 and Q3 INR 297.00. The 1.5×IQR lower and upper fences are INR -283.50 and INR 645.30; there are zero lower outliers and 897 upper outliers. This dataset describes synthetic completed grocery and product orders, with many lower-value purchases and a smaller number of expensive purchases. The mean being much higher than the median and the positive skew show a long right tail. High amounts are outliers by the IQR rule, but some are plausible electronics purchases rather than errors. The largest absolute correlation is order amount with discount percentage (about -0.095); order amount and quantity is about +0.087. Correlation of this size does not establish causation.

## Part B

### RFM and segmentation

RFM uses completed orders through the snapshot `2025-10-01 00:00:00`. There are 1,175 customers with completed orders; recency median is 19 days (minimum 0, maximum 180), frequency mean is 9.46 (maximum 33), and monetary mean is INR 2,552.06 (maximum INR 14,766.55). The script standardises RFM features and tests K-means K=2 through K=8 with `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because it has the strongest silhouette score here, a clear inertia drop from K=3, and useful business profiles. K=8 always has lower inertia than K=2 because more clusters can fit the observations more closely; lower inertia alone rewards making more clusters and does not tell us whether the segments are stable or actionable.

| Segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
|---|---:|---:|---:|---:|---|
| Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Recommend replenishment bundles and simple repeat-order offers |
| Premium High-Value (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | Offer concierge support and relevant premium cross-sells |
| Frequent Essentials | 419 | 8.77 | 19.40 | 3,410.02 | Encourage subscriptions or scheduled replenishment |
| At-Risk / Win-Back | 182 | 109.04 | 2.20 | 685.99 | Send a win-back offer and ask about service friction |

Segment IDs are arbitrary labels, not a ranking. The Premium flag is assigned to the segment with the highest mean monetary value. PCA on scaled RFM explains 63.43% and 21.80% of variance in the first two components (85.23% total). The segment file is `outputs/customer_segments.csv`; metrics and profiles are in `outputs/kmeans_metrics.csv`, `outputs/segment_profile.csv`, and `outputs/pca.json`.

### Lapse prediction

The cutoff is `2025-09-01`. Features use only orders dated before the cutoff; September orders are used only to create the label. The cohort has 1,156 customers, of whom 443 (38.32%) lapsed. The stratified split is 924 training rows and 232 test rows. Segment labels are not features. Numeric scaling and one-hot encoding (with `drop='first'`) are fitted inside a training-only pipeline; the test rows are only transformed. Missing features are filled before fitting, and no feature column contains missing values. The fitted training scaler's mean frequency is 8.115, versus 8.029 over the full cohort, because the training set is only the 80% training sample rather than the full cohort.

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP; FN, TP) | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Always predict not lapsed | — | — | — | — | 0.00 | 0.00 |
| Logistic regression | C=10 | 0.656 | [[104,39],[32,57]] | 0.594 | 0.640 | 0.616 |
| Random forest | max_depth=5, min_samples_leaf=10 | 0.685 | [[109,34],[30,59]] | 0.634 | 0.663 | 0.648 |

The baseline accuracy is 61.6% with lapsed-class F1 of 0. The selected random forest reaches test F1 0.648, above the required 0.55. Missing a lapsing customer means losing a chance to retain them; wrongly targeting a loyal customer wastes an offer and may annoy them. The forest has a useful balance of precision and recall for this demonstration, while the logistic model can be considered when recall is more important.

The five leading random-forest features are frequency (0.367), monetary (0.229), recency (0.093), cancellation rate (0.080), and average discount (0.047). Logistic coefficient examples: frequency is negative (-1.589), suggesting more prior orders reduce predicted lapse odds; recency is positive (+0.034), suggesting a longer gap increases lapse odds; cancellation rate is negative (-0.132) in this fitted model, a dataset-specific relationship that should not be treated as causal. Oversampling the training rows only changes logistic test precision/recall/F1 from 0.594/0.640/0.616 to 0.595/0.876/0.709; recall moves most, helping the retention team catch more likely lapsers but increasing false positives. The saved model reload reproduces test predictions exactly.

`outputs/retention_list.csv` contains up to 10 high-probability test customers flagged Premium; in this run 10 rows were written and 0 actually lapsed. The retention list is illustrative and not a leakage-free evaluation because the Premium flag comes from whole-window RFM, including September, after the lapse cutoff.

## Part C

The pipeline reads tickets in source order. Empty/whitespace tickets are skipped and tickets above 400 estimated tokens are flagged without model calls. Before classification, it masks emails, phone numbers and order IDs, in that order, with `[EMAIL]`, `[PHONE]` and `[ORDER_ID]`. The full batch contains 55 email, 91 phone and 132 order-ID substitutions (278 total); assertions confirm no matching PII patterns remain in redacted prompt text. The user message sent to the mock is exactly the redacted ticket text.

Every model call has a retry wrapper with up to three retries, exponential backoff and jitter (`BASE_DELAY=0.05`). A reply is accepted as JSON only when its category is in the allowed list. If JSON parsing fails, the fallback extracts lowercase alphabetic words and accepts only exactly one allowed category; all other outputs are flagged for human review. No empty or long ticket reaches the model.

| Prompt run | Classified | Accuracy on classified tickets | Total tokens | Tokens / classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 9,400 | 63.95 |
| Few-shot | 147 | 100% (147/147) | 12,988 | 88.35 |

The few-shot prompt uses 3,588 more tokens (about 24.40 extra tokens per classified ticket) but corrects all 21 zero-shot errors in this synthetic run: 11 fraud tickets and 10 technical tickets were incorrectly called billing in the zero-shot run. For a fraud-sensitive support queue, that accuracy gain is worth considering; in a strict low-cost workflow, the zero-shot prompt could be used with conservative human review. Statuses in both runs: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long`, and 4 `skipped_empty`.

The zero-shot confusion counts (actual rows, predicted columns): billing → billing 38, fraud 11, technical 10; delivery → delivery 36; fraud → fraud 23; technical → technical 29. The 12 chatty outputs are accepted with the regex word extractor (so `billing-related` is split correctly); invalid JSON categories such as `shipping` are not guessed and go to human review.

The budget demonstration uses a 6,000-token budget and records 56 `budget_exhausted` tickets; the usage can pass the limit by one call because the rule checks whether the running total is already at or above the budget before the next call. Final few-shot routing is saved in `outputs/triage_results.csv`: fraud goes to high-priority `security_team`; Premium customers otherwise go to high-priority `premium_desk`; valid remaining tickets go to `support_queue`; empty, long, invalid-output and budget-exhausted tickets go to normal-priority `human_review`. Segment-missing customers are treated as non-premium. `outputs/triage_budget_demo.csv` stores the budget run.

## Part D

### Policy RAG

Policy documents are split into 40-word chunks with 10-word overlap: 15 chunks total (refund 4, delivery 3, escalation 3, payment 3, privacy 2). TF measures how often a term appears in a chunk; IDF gives less weight to terms common across many chunks. TF-IDF is lightweight but can miss equivalent phrasings that use different words. Retrieval uses cosine similarity over `TfidfVectorizer` output and returns the top three positive-scoring chunks. The mock assistant is instructed to answer only from supplied context and use a fallback otherwise.

| Chunk overlap | Answerable expected document in top 3 | Expected fact in answer | Unanswerable fallback |
|---|---:|---:|---:|
| 10 words | 12/12 | 12/12 | 2/2 |
| 0 words | 12/12 | 10/12 | 2/2 |

Without overlap, the answer to “What delivery time makes a completed order eligible for a full refund?” and the answer to “How soon is money returned when a payment fails but is deducted?” break because the required phrase is split across adjacent chunks. Results are saved in `outputs/policy_eval_summary.json`.

### Refund agent and safety checks

The five tools are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. The agent checks the SQLite order record and policy retrieval before calculating a refund. Only completed orders qualify; delivery above 45 minutes allows the full amount, delivery of 45 minutes or less gives zero, and amounts above INR 1,000 are escalated rather than issued automatically. All scenarios are treated as within the policy's seven-day request window.

| Scenario | Result | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 907 |
| S2 | REFUND | 135.00 | 5 | 907 |
| S3 | ESCALATE | 1,999.00 | 5 | 894 |
| S4 | DENIED | 0.00 | 4 | 613 |
| S5 | DENIED | 0.00 | 3 | 343 |
| S6 | NOT_FOUND | 0.00 | 2 | 76 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,012 |

S1's token-limit demonstration (`token_limit=200`) ends with `FALLBACK_TOKEN_LIMIT`. The zero-second time-limit demonstration ends with `FALLBACK_TIMEOUT` before the first model call. S7 uses six model calls and ends with `FALLBACK_MAX_ITERATIONS`. Each fallback preserves completed steps and a human-handoff summary.

Crash/resume demonstration: the first two tool executions are checkpointed before a simulated crash. `resume_agent` continues without repeating completed tools. The combined tool sequence is exactly `get_order → search_policy → calculate_refund → issue_refund`; final status is REFUND for INR 128.25. Checkpoints are in `state/`, run records in `outputs/agent_runs.jsonl`, and issue/escalation ledger entries in `outputs/refund_ledger.jsonl`.

**ReAct vs plan-and-execute (summary):** ReAct uses multiple model calls—5 for S1, 3 for S5 and 2 for S6—so it spends more tokens than a plan-and-execute design that asks for a plan once and replans only after an error. However, ReAct sees each real tool observation before deciding the next action, so it can stop correctly on a missing order or cancelled order rather than blindly following a stale plan. For refund decisions, where a wrong action could issue money incorrectly, I prefer ReAct with strict iteration, token and timeout limits plus human handoff. A production version should also add idempotency keys and durable transactional ledger writes.

## Limitations and Next Steps

- This is a synthetic-data learning project, and `mock_llm.py` is deterministic simulation—not a real language model or a measure of production model quality.
- The Premium retention list is illustrative and not a leakage-free evaluation because its flag is derived from whole-window RFM that includes September, even though lapse-model features use only pre-September orders.
- The current model's quality is evaluated on one seeded synthetic split. A real deployment needs time-based validation, monitoring, calibration, privacy/security review and human approval policies.
- TF-IDF is lexical rather than semantic; production retrieval could use embedding search, but this project deliberately stays offline and within the allowed libraries.
- Before production use, add transaction-safe/idempotent refund processing, access controls, audit retention rules, and stronger adversarial prompt/tool validation.

---

## Visual dashboard in Chrome (new)

This project includes a local Streamlit dashboard in `app.py`. It reads the cleaned CSV files and saved Part A–D outputs; it does not connect to live Instamart services or issue real refunds.

### Start the dashboard on Windows

1. Open PowerShell in the `instamart_copilot` project folder.
2. Activate the environment (if PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal first):

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies (including the dashboard libraries):

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. If you have not generated the project outputs yet, run:

   ```powershell
   python run_all.py
   ```

5. Start the dashboard:

   ```powershell
   python -m streamlit run app.py --server.port 8080
   ```

6. Open Chrome at **http://localhost:8080/**. Keep the PowerShell window open while using the dashboard. Press `Ctrl+C` in PowerShell to stop it.

You can also double-click `run_dashboard.bat` after creating `.venv` and installing the requirements. The dashboard has Overview, Orders & Sales, Customers & Churn, Support Tickets, AI Copilot & Policies, and Data Explorer tabs. Use the sidebar filters to change city, order status, and date range. Tables can be downloaded as CSV.
