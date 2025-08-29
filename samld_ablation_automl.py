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

# === 1. 建立 SparkSession（唯一入口）===
spark = SparkSession.builder.appName("SAML-D All-Mode Ablation ML with TimeSeries Split").getOrCreate()
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")  # 防止舊版 datetime 解析錯誤

# === 2. 讀取 parquet 檔案（多模態/原始）===
df_full = spark.read.parquet("gs://saml-d/SAML-D_with_graph_centrality.parquet")  # 多模態/圖論特徵
df_orig = spark.read.parquet("gs://saml-d/SAML-D.parquet")                       # 純原始特徵

# === 3. 統一加 timestamp 欄位（利於時間排序切分）===
def add_timestamp(df):
    return df.withColumn(
        "timestamp",
        unix_timestamp(concat_ws(" ", col("Date"), col("Time")), "yyyy-MM-dd HH:mm:ss").cast("long")
    )

df_full = add_timestamp(df_full)
df_orig = add_timestamp(df_orig)

# === 4. 按 timestamp 嚴格排序，切 80/20 訓練測試 ===
def split_by_time(df):
    df = df.orderBy("timestamp")
    total = df.count()
    split_idx = int(total * 0.8)
    train_data = df.limit(split_idx)
    test_data = df.subtract(train_data)
    return train_data, test_data

train_full, test_full = split_by_time(df_full)
train_orig, test_orig = split_by_time(df_orig)

# === 5. 三種特徵組合（消融實驗分組）===
modes = {
    "多模態(原始+圖論)": {
        "df_train": train_full,
        "df_test": test_full,
        "categorical_cols": [
            "Payment_currency", "Received_currency",
            "Sender_bank_location", "Receiver_bank_location", "Payment_type"
        ],
        "numeric_cols": [
            "Amount",
            "group_node_count", "group_edge_count", "group_bidirect_ratio",
            "sender_degree", "receiver_degree",
            "sender_closeness", "receiver_closeness",
            "sender_betweenness", "receiver_betweenness"
        ]
    },
    "純原始欄位": {
        "df_train": train_orig,
        "df_test": test_orig,
        "categorical_cols": ["Payment_type"],
        "numeric_cols": ["Amount"]
    },
    "純圖論特徵": {
        "df_train": train_full,
        "df_test": test_full,
        "categorical_cols": [],
        "numeric_cols": [
            "group_node_count", "group_edge_count", "group_bidirect_ratio",
            "sender_degree", "receiver_degree",
            "sender_closeness", "receiver_closeness",
            "sender_betweenness", "receiver_betweenness"
        ]
    }
}

# === 6. 定義 ML 經典模型（四種）===
models = {
    "Logistic Regression": LogisticRegression(labelCol="Is_laundering", featuresCol="features"),
    "Decision Tree": DecisionTreeClassifier(labelCol="Is_laundering", featuresCol="features"),
    "Random Forest": RandomForestClassifier(labelCol="Is_laundering", featuresCol="features", numTrees=100),
    "SVM (LinearSVC)": LinearSVC(labelCol="Is_laundering", featuresCol="features")
}

# === 7. 統一指標函數（Weighted, AUC, Support 皆齊全）===
def evaluate_metrics(predictions):
    # ====== ROC AUC & weighted 指標 ======
    auc = BinaryClassificationEvaluator(labelCol="Is_laundering", metricName="areaUnderROC").evaluate(predictions)
    precision_weighted = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", predictionCol="prediction", metricName="weightedPrecision"
    ).evaluate(predictions)
    recall_weighted = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", predictionCol="prediction", metricName="weightedRecall"
    ).evaluate(predictions)
    f1_weighted = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)
    return auc, precision_weighted, recall_weighted, f1_weighted

def eval_class_metrics(predictions, cls):
    # ====== 各類別 Precision/Recall/F1 ======
    prec = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", predictionCol="prediction", metricName="precisionByLabel"
    ).setMetricLabel(cls).evaluate(predictions)
    rec = MulticlassClassificationEvaluator(
        labelCol="Is_laundering", predictionCol="prediction", metricName="recallByLabel"
    ).setMetricLabel(cls).evaluate(predictions)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def get_support(predictions):
    # ====== Support: 各類別樣本數 ======
    return predictions.groupBy("Is_laundering").count().toPandas().set_index("Is_laundering")["count"].to_dict()

# === 8. 主程式執行 ===
for mode_name, cfg in modes.items():
    print("=" * 35)
    print(f"🟩 消融分組：{mode_name}")
    categorical_cols = cfg["categorical_cols"]
    numeric_cols = cfg["numeric_cols"]
    train_data = cfg["df_train"]
    test_data = cfg["df_test"]

    # === 特徵工程（類別編碼/數值拼接）===
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in categorical_cols]
    feature_cols = numeric_cols + [f"{c}_vec" for c in categorical_cols]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    for model_name, clf in models.items():
        print(f"\n🔹【{mode_name} | {model_name}】")
        start = time.time()
        # 無類別特徵就不加編碼器
        stages = indexers + encoders + [assembler, clf] if categorical_cols else [assembler, clf]
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        elapsed = time.time() - start

        # ==== 主要指標 ====
        auc, p_weighted, r_weighted, f1_weighted = evaluate_metrics(predictions)
        print(f"   🕒 訓練+預測時間   ：{elapsed:.2f} 秒")
        print(f"   📈 AUC(ROC)        ：{auc:.4f}")
        print(f"   🎯 Weighted Precision：{p_weighted:.4f}")
        print(f"   🎯 Weighted Recall   ：{r_weighted:.4f}")
        print(f"   🧮 Weighted F1 Score ：{f1_weighted:.4f}")

        # ==== support 指標 ====
        support_dict = get_support(predictions)
        print(f"   🔢 Class Support     ：{support_dict}")

        # ==== 各類別指標 ====
        for cls in [0.0, 1.0]:
            prec, rec, f1 = eval_class_metrics(predictions, cls)
            sup = support_dict.get(cls, 0)
            print(f"   🔹 Class {int(cls)} — Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Support: {sup}")

        # ==== 特徵重要性 or 係數 ====
        if model_name in ["Decision Tree", "Random Forest"]:
            importances = model.stages[-1].featureImportances
            feature_names = numeric_cols.copy()
            for c in categorical_cols:
                ohe_size = 20  # OHE 向量維度，實務可自動抓長度
                feature_names.extend([f"{c}_vec_{i}" for i in range(ohe_size)])
            sorted_features = sorted(zip(feature_names, list(importances)), key=lambda x: -x[1])[:10]
            print("   🔬 Feature Importances:")
            for fname, score in sorted_features:
                print(f"      {fname:<25} {score:.4f}")
        elif model_name == "Logistic Regression":
            coefs = model.stages[-1].coefficients.toArray()
            top_idx = abs(coefs).argsort()[-10:][::-1]
            print("   🔬 Top 10 Coefficients:")
            for idx in top_idx:
                if idx < len(numeric_cols):
                    feat_name = numeric_cols[idx]
                else:
                    feat_name = f"encoded_{idx}"
                print(f"      {feat_name:<25} abs(coef): {abs(coefs[idx]):.4f}")
        elif model_name == "SVM (LinearSVC)":
            coefs = model.stages[-1].coefficients.toArray()
            top_idx = abs(coefs).argsort()[-10:][::-1]
            print("   🔬 Top 10 Coefficients:")
            for idx in top_idx:
                if idx < len(numeric_cols):
                    feat_name = numeric_cols[idx]
                else:
                    feat_name = f"encoded_{idx}"
                print(f"      {feat_name:<25} abs(coef): {abs(coefs[idx]):.4f}")
        print()

print("🎉【三大消融分組，完整主指標+特徵重要性+support指標，全自動比較完畢！】")
