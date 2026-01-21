from collections import defaultdict
import heapq
import math

# ============================================================
# UF-FAE Demo（核心思想畫布）
# ------------------------------------------------------------
# 我們要做的事：
# 1) 原本有一張「帳戶層級」的有向交易圖：u -> v（帶權重 w）
# 2) 另外有一張「實體等價關係」圖：哪些帳戶其實是同一個實體（無向、確定等價）
# 3) 先忽略方向，用 Union-Find 把「同一實體」的帳戶合併成 component（降維）
# 4) 再把原始的「有向交易邊」投影回 component 層級，得到「component-level directed graph」
# 5) 在降維後的有向圖上跑 Dijkstra，算風險/代價最短擴散距離
#
# 重要：Union-Find 是用來合併「確定等價」(equivalence)，
#      Dijkstra 是用來跑「有向傳播/擴散」(directed propagation)。
# ============================================================


# ============================================================
# 1) Union-Find（Disjoint Set Union / DSU）
# ------------------------------------------------------------
# 用途：把「確定等價」的帳戶集合合併成一個 component
# 特性：
# - union(x, y)：表示 x 和 y 屬於同一組（無向、等價）
# - find(x)：找到 x 所在集合的代表（root）
# 優化：
# - path compression（路徑壓縮）
# - union by rank（按秩合併）
# ============================================================
class UnionFind:
    def __init__(self, n: int):
        # parent[i] = i：一開始每個點都是自己的代表
        self.parent = list(range(n))

        # rank[i]：用來避免樹長太高（近似高度）
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        找 x 的集合代表（root）。
        路徑壓縮：讓 x 直接指向 root，讓後續 find 幾乎 O(1)。
        """
        if self.parent[x] != x:
            # 這行做的事：一路找到最上面的 root，再把沿途節點都直接掛到 root
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """
        合併 x 與 y 的集合。
        注意：這裡的 union 表示「等價」，不是交易方向。
        """
        rx, ry = self.find(x), self.find(y)

        # 如果代表一樣，代表早就在同一個集合，不用合併
        if rx == ry:
            return

        # union by rank：把 rank 小的樹接到 rank 大的樹下
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            # rank 相同，隨便接一邊，並讓新 root 的 rank +1
            self.parent[ry] = rx
            self.rank[rx] += 1


# ============================================================
# 2) Dijkstra（Shortest Path on Directed Graph）
# ------------------------------------------------------------
# 用途：在「component-level directed graph」上算最短距離（最小風險成本）
# 前提：邊權重 w 必須是非負（>=0），Dijkstra 才成立
#
# graph 的型態：
#   graph[u] = [(v, w), (v2, w2), ...]
# 表示有向邊 u -> v，權重 w
# ============================================================
def dijkstra(graph: dict, source: int) -> dict:
    """
    回傳 dist：dist[node] = 從 source 到 node 的最短距離
    使用 heapq 當作 priority queue（每次取目前距離最小的點出來擴展）
    """

    # dist 預設為無限大，代表「目前還不知道怎麼到達」
    dist = defaultdict(lambda: math.inf)

    # source 到自己距離是 0
    dist[source] = 0.0

    # 優先佇列：放 (目前距離, 節點)
    pq = [(0.0, source)]

    while pq:
        # 取出目前「最小距離」的節點
        cur_dist, u = heapq.heappop(pq)

        # 如果這個距離已經不是最新（被更短路徑更新過），就跳過
        # 這是 Dijkstra 常見的 lazy deletion 技巧
        if cur_dist > dist[u]:
            continue

        # 嘗試走 u 的每一條外出邊：u -> v（權重 w）
        for v, w in graph[u]:
            # 若走到 v 的新距離更短，更新 dist[v]
            if dist[v] > cur_dist + w:
                dist[v] = cur_dist + w
                heapq.heappush(pq, (dist[v], v))

    return dist


# ============================================================
# 3) UF-FAE Demo Pipeline（完整流程）
# ------------------------------------------------------------
# 這裡把整個「先 UF 無向壓縮 -> 再投影回有向圖 -> Dijkstra」跑一遍
# ============================================================
def uf_fae_demo():
    # --------------------------------------------------------
    # Step A：設定帳戶數（帳戶編號 0 ~ N-1）
    # --------------------------------------------------------
    N = 8

    # --------------------------------------------------------
    # Step B：原始有向交易邊（帳戶層級）
    # transactions 裡每筆是 (u, v, w)
    # 表示有向邊 u -> v，權重 w（可以理解成風險成本/可疑成本/距離）
    #
    # 注意：這裡是「有向」
    # --------------------------------------------------------
    transactions = [
        (0, 2, 3.0),
        (1, 2, 2.0),
        (2, 3, 1.5),
        (3, 4, 2.5),
        (4, 5, 1.0),
        (5, 6, 4.0),
        (6, 7, 1.2),
    ]

    # --------------------------------------------------------
    # Step C：確定等價的實體關係（無向）
    # entity_links 裡每筆是 (a, b)
    # 表示 a 與 b 屬於同一實體（例如同身分、強 KYC 綁定、已人工確認）
    #
    # 注意：這裡是「無向等價」，不代表交易方向
    # --------------------------------------------------------
    entity_links = [
        (0, 1),  # 0 和 1 同一實體
        (5, 6),  # 5 和 6 同一實體
    ]

    # --------------------------------------------------------
    # Step D：Union-Find 壓縮實體
    # --------------------------------------------------------
    uf = UnionFind(N)

    # 把所有等價關係 union 起來
    for u, v in entity_links:
        uf.union(u, v)

    # （可選）如果你想看每個帳戶最後屬於哪個 component，可以印這個：
    # print([uf.find(i) for i in range(N)])

    # --------------------------------------------------------
    # Step E：把原始有向交易邊「投影」到 component 層級
    #
    # component_graph[cu] = [(cv, w), ...]
    # cu = find(u), cv = find(v)
    #
    # 重點：
    # - 方向性 u -> v 保留成 cu -> cv
    # - 若 cu == cv（同一實體內部流動），在「實體間風險擴散」上常可忽略
    #   （因為那是同一個 entity 的內部行為）
    # --------------------------------------------------------
    component_graph = defaultdict(list)

    for u, v, w in transactions:
        cu = uf.find(u)  # u 所屬 component
        cv = uf.find(v)  # v 所屬 component

        # 只保留跨 component 的有向邊
        if cu != cv:
            component_graph[cu].append((cv, w))

    # --------------------------------------------------------
    # Step F：選定風險源頭（source）
    # --------------------------------------------------------
    source_account = 0
    source_component = uf.find(source_account)

    # --------------------------------------------------------
    # Step G：在 component-level directed graph 上跑 Dijkstra
    # --------------------------------------------------------
    dist = dijkstra(component_graph, source_component)

    # --------------------------------------------------------
    # Step H：輸出結果
    # dist[comp] = source_component 到 comp 的最短距離（最小風險成本）
    # --------------------------------------------------------
    print("=== Entity-level directed risk distance ===")
    for comp in sorted(dist.keys()):
        print(f"Component {comp}: {dist[comp]}")


# ============================================================
# 4) 程式進入點
# ============================================================
if __name__ == "__main__":
    uf_fae_demo()
