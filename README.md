# UF-FAE

**Union-Find--based Financial Anomaly Engine**\
*A Graph-Oriented Anti-Money Laundering Framework*

------------------------------------------------------------------------

## 📌 專案簡介（Overview）

**UF-FAE** 是一套以 **Union-Find 圖論演算法** 為核心，結合
**圖結構特徵工程與機器學習** 的反洗錢（AML）分析框架。

本專案的核心思想是：

> **洗錢不是單筆交易問題，而是「群體結構」問題。**

與傳統僅依賴交易欄位（金額、幣別、支付方式）的 AML 方法不同，UF-FAE
將交易網絡視為一個**動態圖系統**，透過近乎常數時間複雜度的 Union-Find
演算法，即時維護帳戶之間的**弱連通群體（WCC）**，並在此基礎上萃取可解釋、可擴展的圖論風險特徵。

------------------------------------------------------------------------

## 📌 Overview (English)

**UF-FAE** is an anti-money laundering (AML) analysis framework built
around the **Union-Find graph algorithm**, integrating
**graph-structured feature engineering** with **machine learning**.

The core premise of this project is:

> **Money laundering is not a single-transaction problem, but a
> group-structure problem.**

Unlike conventional AML approaches that rely primarily on
transaction-level attributes (e.g., amount, currency, payment type),
UF-FAE models financial transactions as a **dynamic graph system**.\
Using Union-Find with near-constant-time complexity, the framework
maintains **weakly connected components (WCCs)** in real time and
extracts interpretable, scalable graph-based risk features on top of
these structures.

------------------------------------------------------------------------

## 🎯 價值主張（Value Proposition）

### 1️⃣ 解決現行 AML 的結構性盲點

-   資料孤島
-   批次處理延遲
-   單筆交易導向
-   高成本且難解釋模型

**UF-FAE 提供以結構與關係為核心的替代方案。**

------------------------------------------------------------------------

### 1️⃣ Addressing Structural Limitations of Existing AML Systems

UF-FAE places graph connectivity and algorithmic efficiency at the
center of AML design.

------------------------------------------------------------------------

## 🧠 核心貢獻（Core Contributions）

### 🔹 Contribution 1

Union-Find 為核心的動態 AML 群體偵測框架（WCC-based）

### 🔹 Contribution 2

可解釋的圖論風險特徵（degree / betweenness / closeness / reciprocity）

### 🔹 Contribution 3

消融實驗證實圖結構對 AML 偵測具決定性影響

### 🔹 Contribution 4

近即時、可擴展至跨銀行與 DLT 環境

------------------------------------------------------------------------

## 📄 Notes

This project is a research-oriented framework validating a
connectivity-centric AML design philosophy.
