# -*- coding: utf-8 -*-
"""
UF-FAE（無 ML）— Polars 版 + 互惠精煉 + 指標面板 + 近似準確性診斷

本程式特色：
1) 流式弱連通 (WCC)：Union-Find（DSU）在邊到達時即時合併。
2) 近似特徵：
   - KMV（K-Minimum Values）：估計群組的去重節點數（distinct）。
   - Count-Min Sketch（CMS）：估計節點的出/入金額（heavy hitters）。
3) 互惠精煉：當群組分數 S 高於門檻，於最近窗口內用「互惠邊 (u→v & v→u)」把 WCC 收緊成有向子塊。
4) Robust z-like 分數：用中位數+MAD 對 distinct / out_hh / density 做穩健標準化，S=0.4*zD+0.4*zHH+0.2*zρ。
5) 指標面板：
   - 定期印出元件數、最大群組規模、最緊群組密度、Top-K 出金重擊者（抽樣）、最近一次合併的 S 明細。
   - 以紅🟥/橘🟧/綠🟩燈顯示風險等級與一行診斷說明。
6) 近似算法準確性指標（理論 + 即時）：
   - KMV：k、樣本數、末秩 R_k、預期相對誤差 ~ 1/sqrt(k)（備註：理論常數略因實作而異）。
   - CMS：w、d、ε=1/w、δ=e^{-d}、當前總流量 N（出、入）、對任意查詢估計的偏差上界 ε*N。

執行前安裝：
  pip install -U polars tqdm
（Windows 建議：設定環境變數 POLARS_MAX_THREADS=16、RAYON_NUM_THREADS=16 以加速 Parquet 解壓，但 Python 主迴圈仍是單執行緒）

⚠️ 注意：
- CMS 因哈希碰撞只「高估不低估」，我們列印的是「可能的上界偏差」供你判讀。
- KMV 的誤差屬於期望與漸近特性；我們給你「理論級」快速指標，不是絕對保證。
"""

# =========================
# 所有 import 都在最前面
# =========================
import os
import csv
import math
import bisect
from typing import Any, Deque, Dict, Tuple, List, Set, Optional
from collections import deque, defaultdict

import polars as pl
from tqdm.auto import tqdm


# =========================
# 使用者可調參數
# =========================

# ---- 檔案與欄位 ----
INPUT_PARQUET = r"C:\Users\Leon\Desktop\程式語言資料\python\UF-FAE\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.parquet"
OUTPUT_ALERTS = "alerts_out.csv"
ASSUME_SORTED = True   # 若 parquet 已按時間排序，True 可省排序與記憶體
TIME_COL = "time"      # 目標時間欄位（會自動嘗試別名）
SENDER_COL = "sender"
RECEIVER_COL = "receiver"
AMOUNT_COL = "amount"

# ---- 近似與視窗設定 ----
KMV_K = 128           # KMV 樣本數（越大越準，記憶體 O(K)）
CMS_W = 4096          # CMS 寬度（誤差 ε ≈ 1/W）
CMS_D = 6             # CMS 深度（錯誤機率 δ ≈ e^{-D}）
WINDOW_EDGES = 2_000_000  # 只保留最近這麼多「已套用」的邊（供精煉觀察）
GAP_SIZE = 10_000         # 延遲處理，吸收亂序（pending queue）

# ---- 判警門檻（robust z-like）----
THRESH_MID = 2.0
THRESH_HIGH = 3.0

# ---- 互惠精煉設定 ----
REFINE_ON_S = True
REFINE_S_THRESH = THRESH_HIGH   # 高於此 S 門檻才觸發精煉
RECIP_WINDOW = 200_000          # 收集最近這麼多條屬於該元件的邊來判互惠
MIN_SUBCOMP_NODES = 2           # 太小的子塊丟掉

# ---- 指標面板與效能微調 ----
PRINT_METRICS_EVERY = 200_000   # 每處理多少邊列印一次面板
RECENT_SENDER_BUFFER = 10_000   # 抽樣重擊者：只掃最近這些 sender
TOPK_HEAVY_HITTERS = 5          # 顯示前 K 名出金重擊者
TQDM_UPDATE_EVERY = 1_000       # 進度條每 1k 筆才更新一次，避免太慢
CSV_FLUSH_EVERY   = 50_000      # 警示累積到 5 萬筆才批次寫檔


# =========================
# 資料結構：DSU / KMV / CMS
# =========================

class DSU:
    """經典 Union-Find；維護弱連通（把圖當無向），近似 O(α(n))。"""
    def __init__(self):
        self.parent: Dict[Any, Any] = {}
        self.rank: Dict[Any, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路徑壓縮
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False, ra, rb
        # 按秩合併
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True, ra, rb

    def reset_subset(self, nodes: Set[Any]):
        """互惠精煉用：把一組節點視為各自獨立，之後再依互惠 union。"""
        for x in nodes:
            self.parent[x] = x
            self.rank[x] = 0


class KMV:
    """
    KMV（K-Minimum Values）去重估計：
    - 只保留 hash 後最小的 K 個值（遞增排序）
    - 估計 distinct ≈ (K-1) * (U / R_k)，U=2^64，R_k=第K小值
    - 預期相對誤差 ~ O(1/sqrt(K))（理論近似）
    """
    def __init__(self, k: int = 64):
        self.k = k
        self.samples: List[int] = []  # 已排序（bisect 插入）

    @staticmethod
    def _hv(x) -> int:
        return hash(x) & 0xffffffffffffffff  # 64-bit 非負

    def add(self, x):
        hv = KMV._hv(x)
        bisect.insort(self.samples, hv)
        if len(self.samples) > self.k:
            self.samples.pop()

    def estimate(self) -> float:
        n = len(self.samples)
        if n < self.k:
            # 樣本不足時，直接以樣本數作為下界
            return float(n)
        U = float(1 << 64)
        r_k = float(self.samples[-1])  # 第 K 小
        if r_k <= 0.0:
            return float(n)
        return (self.k - 1) * (U / r_k)

    # ---- 診斷：KMV 準確性近似 ----
    def diagnostics(self) -> Dict[str, Optional[float]]:
        """
        回傳 KMV 近似準確性指標（理論級快速估計）：
        - k：樣本上限
        - n_samples：目前樣本數（若尚未達 k，代表 distinct 還小）
        - r_k：目前第 k 小值（樣本不足時為 None）
        - rel_error_theory：理論級相對誤差 ~ c / sqrt(k)，這裡用 1/sqrt(k) 做保守估
        """
        n = len(self.samples)
        r_k = float(self.samples[-1]) if n >= self.k else None
        rel_err = 1.0 / math.sqrt(self.k) if self.k > 0 else None
        return {
            "k": float(self.k),
            "n_samples": float(n),
            "r_k": float(r_k) if r_k is not None else None,
            "rel_error_theory": float(rel_err) if rel_err is not None else None,
        }


class CountMinSketch:
    """
    Count-Min Sketch：
    - d 個 hash 函數對應 d 行，w 欄寬；更新與查詢都是 O(d) ≈ O(1)。
    - 查詢取 d 個 bucket 的最小值，為真值上界（只高估不低估）。
    - 誤差上界：f_hat(x) <= f(x) + ε * N，ε ≈ 1/w，機率至少 1-δ（δ ≈ e^{-d}）。
    """
    def __init__(self, width=2048, depth=4, seed=1315423911):
        self.w = width
        self.d = depth
        self.tables = [[0]*self.w for _ in range(self.d)]
        self.seeds = [(seed * (i+1)) & 0xffffffff for i in range(self.d)]

    def _h(self, i, x) -> int:
        # 注意：Python hash 並非 2-universal，這裡做工程近似（如需嚴謹可用 mmh3）
        h = (hash(x) ^ self.seeds[i]) & 0x7fffffff
        return h % self.w

    def update(self, key, val=1):
        v = int(val)
        for i in range(self.d):
            j = self._h(i, key)
            self.tables[i][j] += v

    def query(self, key) -> int:
        est = None
        for i in range(self.d):
            cur = self.tables[i][self._h(i, key)]
            est = cur if est is None else min(est, cur)
        return est or 0


# =========================
# 群組統計：ComponentStats
# =========================

class ComponentStats:
    """對每個 DSU 群組累計 KMV/CMS 與簡易密度、歷史，以供打分與診斷。"""
    def __init__(self, kmv_k=64, cms_w=2048, cms_d=4):
        self.kmv = KMV(k=kmv_k)
        self.cms_out = CountMinSketch(width=cms_w, depth=cms_d)
        self.cms_in  = CountMinSketch(width=cms_w, depth=cms_d)
        self.nodes: Set[Any] = set()
        self.edge_cnt = 0
        self.total_out_flow = 0.0   # 供 CMS 誤差上界使用（N_out）
        self.total_in_flow  = 0.0   # 供 CMS 誤差上界使用（N_in）
        self.history: Deque[Tuple[float,float,float]] = deque(maxlen=256)  # (distinct, out_hh(src), density)

    def update_on_edge(self, u, v, amount):
        """每條邊更新 KMV / CMS / 節點集合 / 邊數。"""
        self.kmv.add(u); self.kmv.add(v)
        self.nodes.add(u); self.nodes.add(v)
        self.edge_cnt += 1
        w = max(1.0, float(amount))  # 防 0 或負數，當作交易強度
        self.cms_out.update(u, val=w)
        self.cms_in.update(v,  val=w)
        self.total_out_flow += w
        self.total_in_flow  += w

    # --- 近似指標 ---
    def approx_distinct(self) -> float:
        return self.kmv.estimate()

    def approx_density(self) -> float:
        n = len(self.nodes)
        if n < 2: return 0.0
        # 無向圖密度近似：2E / (N*(N-1))
        return (2.0 * float(self.edge_cnt)) / (float(n) * float(n-1))

    def approx_out_hh(self, key) -> float:
        return float(self.cms_out.query(key))

    # --- robust z-like ---
    @staticmethod
    def _robust_z(x, series: List[float]) -> float:
        if not series: return 0.0
        data = sorted(series)
        m = data[len(data)//2]
        dev = sorted(abs(xx - m) for xx in data)
        mad = dev[len(dev)//2] if dev else 1.0
        denom = max(1.4826 * mad, 1e-9)
        return (x - m) / denom

    def snapshot_and_score(self, src_key: Any) -> Tuple[float, Dict[str, float]]:
        """擷取當下 3 指標 + z 分數，回傳 S 與明細。"""
        d = self.approx_distinct()
        hh = self.approx_out_hh(src_key)
        dens = self.approx_density()
        z1 = ComponentStats._robust_z(d,   [h[0] for h in self.history])
        z2 = ComponentStats._robust_z(hh,  [h[1] for h in self.history])
        z3 = ComponentStats._robust_z(dens,[h[2] for h in self.history])
        S = 0.4*z1 + 0.4*z2 + 0.2*z3
        self.history.append((d, hh, dens))
        return S, {
            "distinct": d, "out_hh": hh, "density": dens,
            "z_distinct": z1, "z_out_hh": z2, "z_density": z3
        }

    # --- 近似準確性診斷（理論 + 即時）---
    def approximation_diagnostics(self) -> Dict[str, Dict[str, float]]:
        """
        回傳 KMV / CMS 的「理論級」即時診斷：
        KMV：
          - k, n_samples, r_k, rel_error_theory ≈ 1/sqrt(k)
        CMS：
          - w, d, epsilon=1/w, delta=exp(-d)
          - N_out/N_in（總更新值）
          - bound_out = epsilon * N_out（對任何 out 查詢的最大高估偏差）
          - bound_in  = epsilon * N_in
        """
        kmv_diag = self.kmv.diagnostics()
        eps = 1.0 / float(self.cms_out.w)
        delta = math.exp(-float(self.cms_out.d))
        return {
            "kmv": {
                "k": kmv_diag["k"] or 0.0,
                "n_samples": kmv_diag["n_samples"] or 0.0,
                "r_k": kmv_diag["r_k"] if kmv_diag["r_k"] is not None else -1.0,
                "rel_error_theory": kmv_diag["rel_error_theory"] or 0.0,
            },
            "cms": {
                "w": float(self.cms_out.w),
                "d": float(self.cms_out.d),
                "epsilon": eps,
                "delta": delta,
                "N_out": float(self.total_out_flow),
                "N_in":  float(self.total_in_flow),
                "bound_out": eps * float(self.total_out_flow),
                "bound_in":  eps * float(self.total_in_flow),
            }
        }


# =========================
# UF-FAE 引擎：鬆弛→收緊
# =========================

class UF_FAE_RuleEngine:
    """
    主流程：
    - step_edge(): 把 pending 吸收 gap 亂序並實際更新 DSU / 群組統計
    - WCC 合併時計算 S，若高於門檻 → 對該 WCC 做互惠精煉（僅最近窗口）
    - 提供 metrics_summary() 供面板列印（含 KMV/CMS 準確性診斷）
    """
    def __init__(self, kmv_k=64, cms_w=2048, cms_d=4, window_edges=2_000_000, gap_size=0):
        self.dsu = DSU()
        self.comps: Dict[Any, ComponentStats] = {}
        self.pending: Deque[Tuple[float, Any, Any, float]] = deque()
        self.window: Deque[Tuple[float, Any, Any, float]] = deque(maxlen=window_edges)
        self.gap = gap_size
        self.kmv_k, self.cms_w, self.cms_d = kmv_k, cms_w, cms_d

        # 面板需要的一些緩衝
        self.recent_senders: Deque[Any] = deque(maxlen=RECENT_SENDER_BUFFER)
        self.last_merge_metrics: Optional[Dict[str, float]] = None  # 最近合併的 S 與明細

    # ---- 內部工具 ----
    def _comp(self, root):
        if root not in self.comps:
            self.comps[root] = ComponentStats(self.kmv_k, self.cms_w, self.cms_d)
        return self.comps[root]

    def _merge_stats(self, new_root, old_root):
        """Union 之後把 old_root 的統計併到 new_root。"""
        a = self._comp(new_root); b = self.comps.get(old_root)
        if not b: return
        # KMV：把樣本逐一加入（近似）
        for hv in b.kmv.samples: a.kmv.add(hv)
        # CMS：表格逐格相加（行列一致）
        for i in range(a.cms_out.d):
            ai, bi = a.cms_out.tables[i], b.cms_out.tables[i]
            for j in range(a.cms_out.w): ai[j] += bi[j]
            ai2, bi2 = a.cms_in.tables[i], b.cms_in.tables[i]
            for j in range(a.cms_in.w): ai2[j] += bi2[j]
        a.nodes |= b.nodes
        a.edge_cnt += b.edge_cnt
        a.total_out_flow += b.total_out_flow
        a.total_in_flow  += b.total_in_flow
        for h in b.history: a.history.append(h)
        del self.comps[old_root]

    def _update_edge(self, u, v, amount):
        """實際把 u-v 合併、更新統計；回傳 (merged?, root)。"""
        merged, new_root, old_root = self.dsu.union(u, v)
        root = self.dsu.find(u)
        comp = self._comp(root)
        comp.update_on_edge(u, v, amount)
        if merged:
            self._merge_stats(new_root, old_root)
        return merged, root

    # ---- 主推進 ----
    def step_edge(self, t, u, v, amount, thr_mid=2.0, thr_high=3.0):
        """
        進來一條邊 → 先丟 pending；pending 超過 gap → 取出最舊那條實際更新。
        若發生元件合併（merged=True），則計算 S 與警示；必要時做互惠精煉。
        """
        alerts = []
        self.pending.append((t,u,v,amount))
        if len(self.pending) > self.gap:
            t0,u0,v0,a0 = self.pending.popleft()
            merged, root = self._update_edge(u0, v0, a0)
            self.window.append((t0,u0,v0,a0))
            self.recent_senders.append(u0)

            if merged:
                comp = self._comp(root)
                S, d = comp.snapshot_and_score(src_key=u0)

                # 風險等級
                level = "LOW"
                if S >= thr_high: level = "HIGH"
                elif S >= thr_mid: level = "MED"

                # 警示紀錄
                alerts.append({
                    "time": t0, "root": str(root), "src": str(u0), "dst": str(v0),
                    "amount": float(a0), "S": float(round(S,4)), "level": level,
                    "distinct": float(round(d["distinct"],3)),
                    "out_hh":   float(round(d["out_hh"],3)),
                    "density":  float(round(d["density"],6)),
                    "z_distinct": float(round(d["z_distinct"],3)),
                    "z_out_hh":  float(round(d["z_out_hh"],3)),
                    "z_density": float(round(d["z_density"],3)),
                })

                # 最近一次合併明細（面板用）
                self.last_merge_metrics = {
                    "root": str(root),
                    "src": str(u0),
                    "dst": str(v0),
                    "S": float(round(S,4)),
                    "z_distinct": float(round(d["z_distinct"],3)),
                    "z_out_hh": float(round(d["z_out_hh"],3)),
                    "z_density": float(round(d["z_density"],3)),
                    "distinct": float(round(d["distinct"],3)),
                    "out_hh": float(round(d["out_hh"],3)),
                    "density": float(round(d["density"],6)),
                }

                # 觸發互惠精煉（局部）
                try:
                    if REFINE_ON_S and S >= REFINE_S_THRESH:
                        self.refine_component_by_reciprocal(root, window_limit=RECIP_WINDOW)
                except Exception as e:
                    print(f"[REFINE][WARN] {e}")
        return alerts

    def flush_all(self, thr_mid=2.0, thr_high=3.0):
        """把 pending 全部實際更新（結尾清倉）；注意別再 enqueue。"""
        alerts = []
        while self.pending:
            t0, u0, v0, a0 = self.pending.pop()  # 從尾端取：對亂序影響小
            merged, root = self._update_edge(u0, v0, a0)
            self.window.append((t0, u0, v0, a0))
            self.recent_senders.append(u0)

            if merged:
                comp = self._comp(root)
                S, d = comp.snapshot_and_score(src_key=u0)

                level = "LOW"
                if S >= thr_high: level = "HIGH"
                elif S >= thr_mid: level = "MED"

                alerts.append({
                    "time": t0, "root": str(root), "src": str(u0), "dst": str(v0),
                    "amount": float(a0), "S": float(round(S,4)), "level": level,
                    "distinct": float(round(d["distinct"],3)),
                    "out_hh":   float(round(d["out_hh"],3)),
                    "density":  float(round(d["density"],6)),
                    "z_distinct": float(round(d["z_distinct"],3)),
                    "z_out_hh":  float(round(d["z_out_hh"],3)),
                    "z_density": float(round(d["z_density"],3)),
                })

                self.last_merge_metrics = {
                    "root": str(root),
                    "src": str(u0),
                    "dst": str(v0),
                    "S": float(round(S,4)),
                    "z_distinct": float(round(d["z_distinct"],3)),
                    "z_out_hh": float(round(d["z_out_hh"],3)),
                    "z_density": float(round(d["z_density"],3)),
                    "distinct": float(round(d["distinct"],3)),
                    "out_hh": float(round(d["out_hh"],3)),
                    "density": float(round(d["density"],6)),
                }

                try:
                    if REFINE_ON_S and S >= REFINE_S_THRESH:
                        self.refine_component_by_reciprocal(root, window_limit=RECIP_WINDOW)
                except Exception as e:
                    print(f"[REFINE][WARN] {e}")
        return alerts

    # ---- 互惠精煉：WCC（鬆）→ 有向子塊（緊）----
    def refine_component_by_reciprocal(self, root, window_limit=200_000):
        """
        對指定 root 的 WCC 做輕量收緊：
        1) 收集最近 window_limit 筆內屬於此元件的邊。
        2) 以「互惠」(u->v & v->u) 的兩向邊當合併條件，局部重建 DSU。
        3) 依新 DSU 分組，重建 ComponentStats，取代舊 root。
        """
        edges: List[Tuple[Any,Any,float]] = []
        nodes_in_root: Set[Any] = set()
        seen = 0
        for (t,u,v,a) in reversed(self.window):
            if seen >= window_limit:
                break
            if self.dsu.find(u) == root or self.dsu.find(v) == root:
                edges.append((u,v,a))
                nodes_in_root.add(u); nodes_in_root.add(v)
                seen += 1
        if not edges or len(nodes_in_root) < MIN_SUBCOMP_NODES:
            return

        forward = set((u,v) for (u,v,_) in edges)
        reciprocal_pairs = set((u,v) for (u,v) in forward if (v,u) in forward)
        if not reciprocal_pairs:
            return

        # 局部重建
        self.dsu.reset_subset(nodes_in_root)
        for (u,v) in reciprocal_pairs:
            self.dsu.union(u, v)

        # 新分組（僅保留同子塊內的邊）
        groups: Dict[Any, List[Tuple[Any,Any,float]]] = defaultdict(list)
        for (u,v,a) in edges:
            ru = self.dsu.find(u)
            rv = self.dsu.find(v)
            if ru == rv:
                groups[ru].append((u,v,a))

        old_comp = self.comps.get(root)
        if root in self.comps:
            del self.comps[root]

        made = 0
        for sub_root, es in groups.items():
            sub_nodes = set()
            for (u,v,a) in es:
                sub_nodes.add(u); sub_nodes.add(v)
            if len(sub_nodes) < MIN_SUBCOMP_NODES:
                continue
            comp_new = ComponentStats(self.kmv_k, self.cms_w, self.cms_d)
            for (u,v,a) in es:
                comp_new.update_on_edge(u, v, a)
            self.comps[sub_root] = comp_new
            made += 1

        if made == 0 and old_comp is not None:
            # 若沒有合法子塊，避免整個元件消失
            self.comps[root] = old_comp

    # ---- 指標面板：簡潔 + 紅黃綠燈 + 近似準確性 ----
    def metrics_summary(self, topk: int = TOPK_HEAVY_HITTERS) -> Dict[str, Any]:
        """
        回傳面板所需摘要：
        - 元件數、最大群組（distinct~KMV）、最緊群組（密度）
        - 近端 sender 抽樣的 HH(out) Top-K（含可能偏差上界）
        - 最近一次合併的 S 跟 z 分數
        - 近似準確性指標（KMV / CMS）
        """
        # (1) 最大 & 最緊
        num_components = len(self.comps)
        largest_root = None
        largest_distinct = -1.0
        densest_root = None
        largest_density = 0.0

        for r, comp in self.comps.items():
            d = comp.approx_distinct()
            if d > largest_distinct:
                largest_distinct = d
                largest_root = r
            rho = comp.approx_density()
            if rho > largest_density:
                largest_density = rho
                densest_root = r

        # (2) 抽樣 HH（用最近 sender 緩衝）
        uniq = list(dict.fromkeys(self.recent_senders))  # 保序去重
        est_list = []
        for u in uniq:
            root = self.dsu.find(u)
            comp = self.comps.get(root)
            if not comp:
                continue
            est_out = comp.approx_out_hh(u)
            # 對這個「群組」的 CMS 誤差上界（高估）＝ ε * N_out
            eps = 1.0 / float(comp.cms_out.w)
            bound = eps * float(comp.total_out_flow)
            est_list.append((est_out, str(u), str(root), bound))
        est_list.sort(key=lambda x: x[0], reverse=True)
        hh_top = est_list[:topk]

        # (3) 最近一次合併
        lm = self.last_merge_metrics

        # (4) 近似準確性診斷（以「最大群組」為代表；也可以擴展成多群組）
        approx_diag = None
        if largest_root is not None and largest_root in self.comps:
            approx_diag = self.comps[largest_root].approximation_diagnostics()
            approx_diag["represent_root"] = str(largest_root)

        return {
            "num_components": num_components,
            "largest_root": str(largest_root) if largest_root is not None else None,
            "largest_distinct": float(largest_distinct) if largest_distinct >= 0 else None,
            "densest_root": str(densest_root) if densest_root is not None else None,
            "largest_density": float(largest_density),
            "heavy_hitters_top": hh_top,  # list of (est_out, sender, root, bound_over)
            "last_merge": lm,             # dict or None
            "approx_diagnostics": approx_diag,  # KMV / CMS 診斷
        }


# =========================
# Polars 小工具（時間欄位解析）
# =========================

def find_col(names, candidates):
    """在 names 裡找第一個符合 candidates 的欄位（大小寫無關）"""
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def normalize_time_expr(time_col_name: str):
    """
    產生 'tnum'（float 秒），避免昂貴推斷：
      A) 直接數值（float / int）
      B) 純數字字串 → epoch（自動判斷秒/毫秒）
      C) 僅使用明確 format 的 strptime（strict=False）
    全失敗 → null（主迴圈用流水號補）
    """
    t = pl.col(time_col_name)
    t_utf8 = t.cast(pl.Utf8, strict=False)

    numeric = pl.coalesce([
        t.cast(pl.Float64, strict=False),
        t.cast(pl.Int64,   strict=False).cast(pl.Float64, strict=False),
    ])

    digits = (
        t_utf8
        .str.replace_all(r"[^0-9]", "")
        .cast(pl.Float64, strict=False)
    )
    epoch_digits = (
        pl.when((digits.is_not_null()) & (digits > 0))
        .then(pl.when(digits > 1_000_000_000_000.0).then(digits / 1000.0).otherwise(digits))
        .otherwise(None)
        .cast(pl.Float64, strict=False)
    )

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
    ]
    parsed_epochs = [
        t_utf8
        .str.strptime(pl.Datetime, format=fmt, strict=False, exact=False)
        .dt.epoch(time_unit="s")
        .cast(pl.Float64, strict=False)
        for fmt in fmts
    ]

    return pl.coalesce([numeric, epoch_digits, *parsed_epochs]).alias("tnum")


def row_iter():
    """
    回傳逐列迭代器：(tnum, sender, receiver, amount)
    - 自動偵測實際欄位名（大小寫無關 + 常見別名）
    - ASSUME_SORTED=True：engine="streaming"（低記憶體）
      否則：sort 再 collect（engine="auto"）
    """
    lf0 = pl.scan_parquet(INPUT_PARQUET)
    schema = lf0.collect_schema()
    names = list(schema.keys())

    time_actual = find_col(names, [
        TIME_COL, "time", "Time", "timestamp", "Timestamp",
        "datetime", "Datetime", "date", "Date"
    ])
    sender_actual = find_col(names, [
        SENDER_COL, "sender", "Sender", "sender_account", "Sender_account",
        "src", "from", "From", "payer", "Payer"
    ])
    receiver_actual = find_col(names, [
        RECEIVER_COL, "receiver", "Receiver", "receiver_account", "Receiver_account",
        "dst", "to", "To", "payee", "Payee"
    ])
    amount_actual = find_col(names, [
        AMOUNT_COL, "amount", "Amount", "value", "Value",
        "payment_amount", "Payment_amount", "amt", "Amt"
    ])

    missing = [k for k, v in [
        ("time", time_actual),
        ("sender", sender_actual),
        ("receiver", receiver_actual),
        ("amount", amount_actual),
    ] if v is None]
    if missing:
        print("[SCHEMA] 檔案欄位：", names)
        raise RuntimeError(f"[SCHEMA] 找不到必要欄位：{missing}；請調整別名或檔案欄位。")

    lf = lf0.select([
        normalize_time_expr(time_actual),
        pl.col(sender_actual).alias("sender"),
        pl.col(receiver_actual).alias("receiver"),
        pl.col(amount_actual).cast(pl.Float64, strict=False).alias("amount"),
    ])

    if ASSUME_SORTED:
        df = lf.collect(engine="streaming")
    else:
        df = lf.sort("tnum").collect(engine="auto")

    return df.iter_rows(named=True)


# =========================
# 主程式：串流處理 + 面板 + 近似診斷
# =========================

def _level_emoji(level: str) -> str:
    """把等級轉紅黃綠燈 emoji。"""
    return {"HIGH":"🟥", "MED":"🟧", "LOW":"🟩"}.get(level, "⬜")

def _short_dx(zD, zHH, zRho) -> str:
    """一句話診斷：哪些維度在拉警報。"""
    parts = []
    if zD >= 2.0: parts.append("規模擴張")
    if zHH >= 2.0: parts.append("出金集中")
    if zRho >= 2.0: parts.append("群內緊密")
    if not parts: return "平穩"
    if len(parts) == 1: return f"{parts[0]}"
    if len(parts) == 2: return f"{parts[0]} + {parts[1]}"
    return "規模+集中+緊密（高度可疑）"

def print_metrics(engine: UF_FAE_RuleEngine, alerted: int, cnt: int):
    """人讀友善的面板，包含近似準確性指標。"""
    metrics = engine.metrics_summary()
    hh_str = ", ".join([
        f"{i+1}:{sender}/root={root}≈{int(val)} (≤+{int(bound)})"
        for i,(val,sender,root,bound) in enumerate(metrics["heavy_hitters_top"])
    ])

    print(f"[PROGRESS] edges={cnt:,} alerts={alerted:,}")
    print(
        "[METRICS] comps={:,} | largest(V)={}@{} | densest={} ρ={:.6f} | HH(out) [{}]".format(
            metrics["num_components"],
            int(metrics["largest_distinct"] or 0), metrics["largest_root"],
            metrics["densest_root"], metrics["largest_density"],
            hh_str,
        )
    )

    # 最近一次合併的分數與診斷
    if metrics["last_merge"]:
        lm = metrics["last_merge"]
        level = "HIGH" if lm["S"] >= THRESH_HIGH else ("MED" if lm["S"] >= THRESH_MID else "LOW")
        print(
            "          [LAST-MERGE] {} root={} src={} dst={} | S={:.3f} | "
            "zD={:.2f} zHH={:.2f} zρ={:.2f} | D~{} HH~{} ρ~{:.6f} | 診斷: {}".format(
                _level_emoji(level),
                lm["root"], lm["src"], lm["dst"], lm["S"],
                lm["z_distinct"], lm["z_out_hh"], lm["z_density"],
                lm["distinct"], lm["out_hh"], lm["density"],
                _short_dx(lm["z_distinct"], lm["z_out_hh"], lm["z_density"])
            )
        )

    # 近似診斷（以最大群組為代表）
    if metrics["approx_diagnostics"]:
        ad = metrics["approx_diagnostics"]
        kmv = ad["kmv"]; cms = ad["cms"]
        print(
            "          [APPROX] root={} | KMV: k={} samples={} r_k={} 期望相對誤差≈{:.3f} | "
            "CMS: w={} d={} ε=1/w≈{:.6f} δ≈e^-d≈{:.6f} | N_out≈{:.0f} → 任何估值高估上界 ≤ ε*N_out≈{:.0f}".format(
                ad.get("represent_root","?"),
                int(kmv["k"]), int(kmv["n_samples"]), int(kmv["r_k"]) if kmv["r_k"]>=0 else "NA",
                float(kmv["rel_error_theory"]),
                int(cms["w"]), int(cms["d"]), float(cms["epsilon"]), float(cms["delta"]),
                float(cms["N_out"]), float(cms["bound_out"])
            )
        )

def main():
    print("[INFO] reading:", INPUT_PARQUET)

    # 可選：若要強制使用 16 執行緒（Polars/Rayon），可在外部設定環境變數後再執行。
    os.environ.setdefault("POLARS_MAX_THREADS", "16")
    os.environ.setdefault("RAYON_NUM_THREADS", "16")

    # 嘗試取得總列數以美化進度條；失敗就用不定長度
    try:
        total_edges = pl.scan_parquet(INPUT_PARQUET).select(pl.count()).collect().item()
    except Exception:
        total_edges = None

    engine = UF_FAE_RuleEngine(
        kmv_k=KMV_K, cms_w=CMS_W, cms_d=CMS_D,
        window_edges=WINDOW_EDGES, gap_size=GAP_SIZE
    )

    header = ["time","root","src","dst","amount","S","level",
              "distinct","out_hh","density","z_distinct","z_out_hh","z_density"]
    if os.path.exists(OUTPUT_ALERTS):
        os.remove(OUTPUT_ALERTS)
    f = open(OUTPUT_ALERTS, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=header); w.writeheader()

    cnt = 0
    alerted = 0
    pbar = tqdm(total=total_edges, unit="edge", dynamic_ncols=True, desc="UF-FAE streaming")
    alerts_buffer: List[Dict[str, Any]] = []

    for r in row_iter():
        # 取欄位（tnum 失敗時以流水號補）
        t = float(r["tnum"]) if r["tnum"] is not None else float(cnt)
        u = r["sender"]; v = r["receiver"]
        amt = float(r["amount"]) if r["amount"] is not None else 0.0

        # 主推進
        alerts = engine.step_edge(t,u,v,amt, thr_mid=THRESH_MID, thr_high=THRESH_HIGH)
        if alerts:
            alerts_buffer.extend(alerts)
            alerted += len(alerts)
            # 批次寫檔，減少 I/O 次數
            if len(alerts_buffer) >= CSV_FLUSH_EVERY:
                w.writerows(alerts_buffer)
                alerts_buffer.clear()

        # 進度與面板節流
        cnt += 1
        if cnt % TQDM_UPDATE_EVERY == 0:
            pbar.update(TQDM_UPDATE_EVERY)

        if cnt % PRINT_METRICS_EVERY == 0:
            pbar.set_postfix_str(f"alerts={alerted:,} comps={engine.metrics_summary()['num_components']}")
            print_metrics(engine, alerted, cnt)

    # 收尾：處理殘餘 pending
    tail_alerts = engine.flush_all(thr_mid=THRESH_MID, thr_high=THRESH_HIGH)
    if tail_alerts:
        alerts_buffer.extend(tail_alerts)
        alerted += len(tail_alerts)

    # 把緩衝中的警示一次寫出
    if alerts_buffer:
        w.writerows(alerts_buffer)
        alerts_buffer.clear()

    f.flush(); f.close()
    print_metrics(engine, alerted, cnt)
    print(f"[DONE] total edges={cnt:,}, alerts={alerted:,}, output -> {OUTPUT_ALERTS}")


if __name__ == "__main__":
    main()
