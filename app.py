"""
VectorDB Python Backend — drop-in replacement for main.cpp
Run: uvicorn main:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import math
import time
import threading
import heapq
import random
from pathlib import Path
from typing import Optional, Callable

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DIMS = 16
DistFn = Callable[[list[float], list[float]], float]


# ─────────────────────────────────────────────
#  DISTANCE METRICS
# ─────────────────────────────────────────────
def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    return {"cosine": cosine, "manhattan": manhattan}.get(metric, euclidean)


# ─────────────────────────────────────────────
#  VECTOR ITEM
# ─────────────────────────────────────────────
class VectorItem:
    __slots__ = ("id", "metadata", "category", "emb")

    def __init__(self, id: int, metadata: str, category: str, emb: list[float]):
        self.id       = id
        self.metadata = metadata
        self.category = category
        self.emb      = emb


# ─────────────────────────────────────────────
#  BRUTE FORCE
# ─────────────────────────────────────────────
class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        scored = sorted((dist(q, v.emb), v.id) for v in self.items)
        return scored[:k]


# ─────────────────────────────────────────────
#  KD-TREE
# ─────────────────────────────────────────────
class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item  = item
        self.left:  Optional[_KDNode] = None
        self.right: Optional[_KDNode] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[_KDNode] = None

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _insert(self, node: Optional[_KDNode], v: VectorItem, depth: int) -> _KDNode:
        if node is None:
            return _KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        # max-heap stored as (-dist, id)
        heap: list[tuple[float, int]] = []
        self._search(self.root, q, k, 0, dist, heap)
        return sorted((-d, id) for d, id in heap)

    def _search(self, node: Optional[_KDNode], q: list[float], k: int,
                depth: int, dist: DistFn, heap: list):
        if node is None:
            return
        d = dist(q, node.item.emb)
        if len(heap) < k or d < -heap[0][0]:
            heapq.heappush(heap, (-d, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        near = node.left  if diff < 0 else node.right
        far  = node.right if diff < 0 else node.left

        self._search(near, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._search(far, q, k, depth + 1, dist, heap)


# ─────────────────────────────────────────────
#  HNSW
# ─────────────────────────────────────────────
class _HNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item: VectorItem, max_lyr: int):
        self.item    = item
        self.max_lyr = max_lyr
        self.nbrs: list[list[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M        = M
        self.M0       = 2 * M
        self.ef_build = ef_build
        self.mL       = 1.0 / math.log(float(M))
        self.top_lyr  = -1
        self.entry    = -1
        self.G: dict[int, _HNode] = {}
        self._rng     = random.Random(42)

    def _rand_lvl(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      lyr: int, dist: DistFn) -> list[tuple[float, int]]:
        if ep not in self.G:
            return []
        d0  = dist(q, self.G[ep].item.emb)
        vis = {ep}

        # cands = min-heap, found = max-heap (negated)
        cands: list[tuple[float, int]] = [(d0, ep)]
        found: list[tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst   = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = sorted((-d, id) for d, id in found)
        return result

    def insert(self, item: VectorItem, dist: DistFn):
        id  = item.id
        lvl = self._rand_lvl()
        self.G[id] = _HNode(item, lvl)

        if self.entry == -1:
            self.entry   = id
            self.top_lyr = lvl
            return

        ep = self.entry
        for lc in range(self.top_lyr, lvl, -1):
            if ep in self.G and lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_lyr, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel  = [w[1] for w in W[:maxM]]
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                nn = self.G[nid]
                while lc >= len(nn.nbrs):
                    nn.nbrs.append([])
                nn.nbrs[lc].append(id)
                if len(nn.nbrs[lc]) > maxM:
                    ds = sorted(
                        (dist(nn.item.emb, self.G[c].item.emb), c)
                        for c in nn.nbrs[lc] if c in self.G
                    )
                    nn.nbrs[lc] = [c for _, c in ds[:maxM]]

            if W:
                ep = W[0][1]

        if lvl > self.top_lyr:
            self.top_lyr = lvl
            self.entry   = id

    def knn(self, q: list[float], k: int, ef: int,
            dist: DistFn) -> list[tuple[float, int]]:
        if self.entry == -1:
            return []
        ep = self.entry
        for lc in range(self.top_lyr, 0, -1):
            if ep in self.G and lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G:
            return
        for nd in self.G.values():
            for layer in nd.nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry == id:
            self.entry = next((nid for nid in self.G if nid != id), -1)
        del self.G[id]

    def get_info(self) -> dict:
        max_l = max(self.top_lyr + 1, 1)
        npl   = [0] * max_l
        epl   = [0] * max_l
        nodes, edges = [], []

        for id, nd in self.G.items():
            nodes.append({"id": id, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_lyr})
            for lc in range(min(nd.max_lyr + 1, max_l)):
                npl[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            epl[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})

        return {"topLayer": self.top_lyr, "nodeCount": len(self.G),
                "nodesPerLayer": npl, "edgesPerLayer": epl,
                "nodes": nodes, "edges": edges}

    def __len__(self):
        return len(self.G)


# ─────────────────────────────────────────────
#  VECTOR DB  (16-D demo index)
# ─────────────────────────────────────────────
class VectorDB:
    def __init__(self, dims: int):
        self.dims  = dims
        self.store: dict[int, VectorItem] = {}
        self.bf    = BruteForce()
        self.kdt   = KDTree(dims)
        self.hnsw  = HNSW(16, 200)
        self._lock = threading.Lock()
        self._nid  = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(self._nid, meta, cat, emb)
            self._nid += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str, algo: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id in raw:
                if id in self.store:
                    v = self.store[id]
                    hits.append({"id": v.id, "metadata": v.metadata,
                                 "category": v.category, "embedding": v.emb,
                                 "distance": d})
            return {"hits": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)

            def t(fn):
                s = time.perf_counter()
                fn()
                return int((time.perf_counter() - s) * 1_000_000)

            return {
                "bruteforceUs": t(lambda: self.bf.knn(q, k, dfn)),
                "kdtreeUs":     t(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs":       t(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "itemCount":    len(self.store),
            }

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self.store)


# ─────────────────────────────────────────────
#  TEXT CHUNKER
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_words: int = 250, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    step, chunks, i = chunk_words - overlap, [], 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# ─────────────────────────────────────────────
#  OLLAMA CLIENT
# ─────────────────────────────────────────────
class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base        = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = httpx.post(f"{self.base}/api/embeddings",
                           json={"model": self.embed_model, "prompt": text},
                           timeout=30)
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = httpx.post(f"{self.base}/api/generate",
                           json={"model": self.gen_model,
                                 "prompt": prompt, "stream": False},
                           timeout=180)
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# ─────────────────────────────────────────────
#  DOCUMENT DB
# ─────────────────────────────────────────────
class DocItem:
    __slots__ = ("id", "title", "text", "emb")

    def __init__(self, id: int, title: str, text: str, emb: list[float]):
        self.id, self.title, self.text, self.emb = id, title, text, emb


class DocumentDB:
    def __init__(self):
        self.store: dict[int, DocItem] = {}
        self.hnsw  = HNSW(16, 200)
        self.bf    = BruteForce()
        self._lock = threading.Lock()
        self._nid  = 1
        self.dims  = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self._lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self._nid, title, text, emb)
            self._nid += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self.store:
                return []
            raw = (self.bf.knn(q, k, cosine)
                   if len(self.store) < 10
                   else self.hnsw.knn(q, k, 50, cosine))
            return [(d, self.store[id])
                    for d, id in raw
                    if id in self.store and d <= max_dist]

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self.store.values())

    def size(self) -> int:
        with self._lock:
            return len(self.store)


# ─────────────────────────────────────────────
#  DEMO DATA
# ─────────────────────────────────────────────
DEMO: list[tuple[str, str, list[float]]] = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]


def load_demo(db: VectorDB):
    dfn = get_dist_fn("cosine")
    for meta, cat, emb in DEMO:
        db.insert(meta, cat, emb, dfn)


# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="VectorDB Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_db     = VectorDB(DIMS)
_doc_db = DocumentDB()
_ollama = OllamaClient()

load_demo(_db)

_ollama_up = _ollama.is_available()
print("=== VectorDB Engine (Python) ===")
print(f"http://localhost:8080")
print(f"{_db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
print(f"Ollama: {'ONLINE' if _ollama_up else 'OFFLINE (install from ollama.com)'}")


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def parse_vec(s: str) -> list[float]:
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return []


# ─────────────────────────────────────────────
#  FAVICON  (suppress 404 noise)
# ─────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


# ─────────────────────────────────────────────
#  SERVE index.html
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    p = Path("index.html")
    if not p.exists():
        raise HTTPException(404, "index.html not found — place it next to main.py")
    return HTMLResponse(p.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
#  DEMO VECTOR ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    q = parse_vec(v)
    if len(q) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-D vector, got {len(q)}")
    out = _db.search(q, k, metric, algo)
    return {
        "results": [
            {
                "id":        h["id"],
                "metadata":  h["metadata"],
                "category":  h["category"],
                "distance":  round(h["distance"], 6),
                "embedding": h["embedding"],
            }
            for h in out["hits"]
        ],
        "latencyUs": out["latencyUs"],
        "algo":      out["algo"],
        "metric":    out["metric"],
    }


@app.get("/items")
def items():
    return [
        {"id": v.id, "metadata": v.metadata,
         "category": v.category, "embedding": v.emb}
        for v in _db.all()
    ]


@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    q = parse_vec(v)
    if len(q) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-D vector, got {len(q)}")
    b = _db.benchmark(q, k, metric)
    return {"bruteforceUs": b["bruteforceUs"], "kdtreeUs": b["kdtreeUs"],
            "hnswUs": b["hnswUs"], "itemCount": b["itemCount"]}


@app.get("/hnsw-info")
def hnsw_info():
    return _db.hnsw_info()


@app.get("/stats")
def stats():
    return {"count": _db.size(), "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics":    ["euclidean", "cosine", "manhattan"]}


@app.get("/status")
def status():
    up = _ollama.is_available()
    return {"ollamaAvailable": up,
            "embedModel":      _ollama.embed_model,
            "genModel":        _ollama.gen_model,
            "docCount":        _doc_db.size(),
            "docDims":         _doc_db.dims,
            "demoDims":        DIMS,
            "demoCount":       _db.size()}


# ─────────────────────────────────────────────
#  INSERT / DELETE  (Pydantic bodies)
# ─────────────────────────────────────────────
class InsertBody(BaseModel):
    metadata:  str
    category:  str
    embedding: list[float]


@app.post("/insert")
def insert(body: InsertBody):
    if len(body.embedding) != DIMS:
        raise HTTPException(400, f"Expected {DIMS}-D embedding")
    id = _db.insert(body.metadata, body.category,
                    body.embedding, get_dist_fn("cosine"))
    return {"id": id}


@app.delete("/delete/{id}")
def delete(id: int):
    return {"ok": _db.remove(id)}


# ─────────────────────────────────────────────
#  DOCUMENT ENDPOINTS
# ─────────────────────────────────────────────
class DocInsertBody(BaseModel):
    title: str
    text:  str


@app.post("/doc/insert")
def doc_insert(body: DocInsertBody):
    if not body.title.strip() or not body.text.strip():
        raise HTTPException(400, "Need both title and text")

    chunks = chunk_text(body.text, 250, 30)
    ids    = []

    for i, chunk in enumerate(chunks):
        emb = _ollama.embed(chunk)
        if not emb:
            return JSONResponse(status_code=503, content={
                "error": (
                    "Ollama unavailable. Install from https://ollama.com then run:\n"
                    "  ollama pull nomic-embed-text\n"
                    "  ollama pull llama3.2"
                )
            })
        title = (f"{body.title} [{i+1}/{len(chunks)}]"
                 if len(chunks) > 1 else body.title)
        ids.append(_doc_db.insert(title, chunk, emb))

    return {"ids": ids, "chunks": len(chunks), "dims": _doc_db.dims}


@app.get("/doc/list")
def doc_list():
    return [
        {
            "id":      d.id,
            "title":   d.title,
            "preview": d.text[:120] + ("…" if len(d.text) > 120 else ""),
            "words":   len(d.text.split()),
        }
        for d in _doc_db.all()
    ]


@app.delete("/doc/delete/{id}")
def doc_delete(id: int):
    return {"ok": _doc_db.remove(id)}


class DocSearchBody(BaseModel):
    question: str
    k: int = 3


@app.post("/doc/search")
def doc_search(body: DocSearchBody):
    if not body.question.strip():
        raise HTTPException(400, "Need a question")
    emb = _ollama.embed(body.question)
    if not emb:
        return JSONResponse(status_code=503,
                            content={"error": "Ollama unavailable"})
    hits = _doc_db.search(emb, body.k)
    return {
        "contexts": [
            {"id": d.id, "title": d.title, "distance": round(dist, 4)}
            for dist, d in hits
        ]
    }


class AskBody(BaseModel):
    question: str
    k: int = 3


@app.post("/doc/ask")
def doc_ask(body: AskBody):
    if not body.question.strip():
        raise HTTPException(400, "Need a question")

    # 1. Embed question
    emb = _ollama.embed(body.question)
    if not emb:
        return JSONResponse(status_code=503,
                            content={"error": "Ollama unavailable"})

    # 2. Retrieve context
    hits = _doc_db.search(emb, body.k)

    # 3. Build prompt
    ctx = "\n\n".join(
        f"[{i+1}] {d.title}:\n{d.text}"
        for i, (_, d) in enumerate(hits)
    )
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, use your own general knowledge. "
        "Do NOT mention 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer naturally.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {body.question}\n\nAnswer:"
    )

    # 4. Generate
    answer = _ollama.generate(prompt)

    # 5. Return
    return {
        "answer":   answer,
        "model":    _ollama.gen_model,
        "contexts": [
            {"id": d.id, "title": d.title,
             "text": d.text, "distance": round(dist, 4)}
            for dist, d in hits
        ],
        "docCount": _doc_db.size(),
    }
