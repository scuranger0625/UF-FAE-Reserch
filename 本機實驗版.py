import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, concat_ws, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, LinearSVC
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# ========== 1. SparkSession ==========
spark = (
    SparkSession.builder
    .appName("SAML-D All Baseline Modes with TimeSeries Split")
    .config("spark.driver.memory", "30g")
    .config("spark.executor.memory", "20g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

# ========== 2. 讀取資料 ==========
df_orig = spark.read.parquet(
    r"C:\Users\Leon\Desktop\程式語言資料\python\UF-FAE\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.parquet"
)

df_graph = spark.read.parquet(
    r"C:\Users\Leon\Desktop\程式語言資料\python\UF-FAE\Anti Money Laundering Transaction Data (SAML-D)\SAML-D_with_graph_centrality.parquet"
)

# ========== 3. 加 timestamp ==========
def add_ts(df):
    return df.withColumn(
        "timestamp",
        unix_timestamp(concat_ws(" ", col("Date"), col("Time")), "yyyy-MM-dd HH:mm:ss").cast("long")
    )

df_orig = add_ts(df_orig)
df_graph = add_ts(df_graph)

# ========== 4. 時間切分 ==========
df_orig = df_orig.orderBy("timestamp")
df_graph = df_graph.orderBy("timestamp")

total = df_orig.count()
split_idx = int(total * 0.8)
boundary_ts = df_orig.select("timestamp").take(split_idx)[-1][0]

train_orig = df_orig.filter(col("timestamp") <= boundary_ts)
test_orig  = df_orig.filter(col("timestamp") > boundary_ts)

train_graph = df_graph.filter(col("timestamp") <= boundary_ts)
test_graph  = df_graph.filter(col("timestamp") > boundary_ts)

# ========== 5. Baseline 三種模式 ==========
modes = {
    "純原生欄位": {
        "train": train_orig,
        "test": test_orig,
        "categorical": [
            "Payment_currency",
            "Received_currency",
            "Sender_bank_location",
            "Receiver_bank_location",
            "Payment_type"
        ],
        "numeric": ["Amount"]
    },
    "純圖論欄位": {
        "train": train_graph,
        "test": test_graph,
        "categorical": [],
        "numeric": [
            "group_node_count", "group_edge_count", "group_bidirect_ratio",
            "sender_degree", "receiver_degree",
            "sender_closeness", "receiver_closeness",
            "sender_betweenness", "receiver_betweenness"
        ]
    },
    "多模態（原生+圖論）": {
        "train": train_graph,
        "test": test_graph,
        "categorical": [
            "Payment_currency",
            "Received_currency",
            "Sender_bank_location",
            "Receiver_bank_location",
            "Payment_type"
        ],
        "numeric": [
            "Amount",
            "group_node_count", "group_edge_count", "group_bidirect_ratio",
            "sender_degree", "receiver_degree",
            "sender_closeness", "receiver_closeness",
            "sender_betweenness", "receiver_betweenness"
        ]
    }
}

# ========== 6. 模型 ==========
models = {
    "Logistic Regression": LogisticRegression(labelCol="Is_laundering", featuresCol="features"),
    "Decision Tree": DecisionTreeClassifier(labelCol="Is_laundering", featuresCol="features"),
    "Random Forest": RandomForestClassifier(labelCol="Is_laundering", featuresCol="features", numTrees=100),
    "SVM (LinearSVC)": LinearSVC(labelCol="Is_laundering", featuresCol="features")
}

# ========== 7. 指標 ==========
def evaluate_metrics(pred):
    auc = BinaryClassificationEvaluator(labelCol="Is_laundering", metricName="areaUnderROC").evaluate(pred)
    p = MulticlassClassificationEvaluator(labelCol="Is_laundering", metricName="weightedPrecision").evaluate(pred)
    r = MulticlassClassificationEvaluator(labelCol="Is_laundering", metricName="weightedRecall").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(labelCol="Is_laundering", metricName="f1").evaluate(pred)
    return auc, p, r, f1

def eval_cls(pred, cls):
    prec = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", metricName="precisionByLabel"
    ).setMetricLabel(cls).evaluate(pred)

    rec = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", metricName="recallByLabel"
    ).setMetricLabel(cls).evaluate(pred)

    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
    return prec, rec, f1


# ========== 8. 主迴圈（自動跑三種 Baseline）==========
print("\n=======================================")
print("🔥【三大 Baseline 模式開始 — Ablation Study】")
print("=======================================\n")

for mode_name, cfg in modes.items():

    print("\n=======================================")
    print(f"🟩 模式：{mode_name}")
    print("=======================================\n")

    train_df = cfg["train"]
    test_df = cfg["test"]
    cat_cols = cfg["categorical"]
    num_cols = cfg["numeric"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in cat_cols]
    feature_cols = num_cols + [f"{c}_vec" for c in cat_cols]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    for model_name, clf in models.items():
        print(f"\n🔹【{mode_name} | {model_name}】")

        stages = indexers + encoders + [assembler, clf] if cat_cols else [assembler, clf]
        pipeline = Pipeline(stages=stages)

        start = time.time()
        model = pipeline.fit(train_df)
        preds = model.transform(test_df)
        elapsed = time.time() - start

        auc, p_w, r_w, f1_w = evaluate_metrics(preds)

        print(f"   🕒 訓練+預測時間   ：{elapsed:.2f} 秒")
        print(f"   📈 AUC(ROC)        ：{auc:.4f}")
        print(f"   🎯 Weighted Precision：{p_w:.4f}")
        print(f"   🎯 Weighted Recall   ：{r_w:.4f}")
        print(f"   🧮 Weighted F1 Score ：{f1_w:.4f}")

        for cls in [0.0, 1.0]:
            pr, rc, f1 = eval_cls(preds, cls)
            print(f"   🔹 Class {int(cls)} — Precision: {pr:.4f}, Recall: {rc:.4f}, F1: {f1:.4f}")

print("\n🎉【三大 Baseline 全部完成 — 可直接對照 UF-FAE】")
