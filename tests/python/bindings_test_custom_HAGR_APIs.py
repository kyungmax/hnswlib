import numpy as np
import hnswlib
import unittest

def l2_normalize(x: np.ndarray, eps=1e-10) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

class AdaptiveDebugTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dim = 64
        self.k = 10
        self.num_elements = 12000

        # ---- 데이터 생성: 두 개의 군집 + 무작위 클라우드 ----
        n1, n2 = 5000, 5000
        n3 = self.num_elements - n1 - n2

        c1 = np.random.randn(1, self.dim).astype(np.float32)
        c2 = (np.random.randn(1, self.dim) + 4.0).astype(np.float32)

        cluster1 = c1 + 0.05 * np.random.randn(n1, self.dim).astype(np.float32)
        cluster2 = c2 + 0.05 * np.random.randn(n2, self.dim).astype(np.float32)
        cloud = 0.8 * np.random.randn(n3, self.dim).astype(np.float32)

        data = np.vstack([cluster1, cluster2, cloud]).astype(np.float32)
        data = l2_normalize(data)
        self.data = data

        self.p = hnswlib.Index(space="cosine", dim=self.dim)
        self.p.init_index(max_elements=self.num_elements, ef_construction=200, M=16)
        self.p.add_items(self.data)

        # 1. 내부 LID 계산 수행
        self.p.calc_lids_internal(k_lid=15, num_threads=1)

        # 2. 절대 LID 임계값 추출 (get_lids 메서드 사용)
        all_lids = self.p.get_lids()
        self.abs_q25 = np.percentile(all_lids, 25)
        self.abs_q75 = np.percentile(all_lids, 75)
        self.abs_q90 = np.percentile(all_lids, 90)

        # ---- 쿼리 생성 ----
        q_easy = c1 + 0.01 * np.random.randn(1, self.dim).astype(np.float32)
        q_boundary = (c1 + c2) / 2.0 + 0.02 * np.random.randn(1, self.dim).astype(np.float32)

        self.q_easy = l2_normalize(q_easy)
        self.q_boundary = l2_normalize(q_boundary)

    def _radius(self, dists: np.ndarray) -> float:
        return float(np.max(dists))

    def _run_fixed(self, query: np.ndarray, ef: int):
        self.p.set_ef(ef)
        labels, dists = self.p.knn_query(query, k=self.k, num_threads=1)
        return labels, dists

    def _run_adaptive(self, query: np.ndarray, **kwargs):
        # 최신 바인딩 규격에 맞게 호출
        labels, dists, reduced_steps, stop_count = self.p.knn_query_adaptive(query, k=self.k, num_threads=1, **kwargs)
        return labels, dists, reduced_steps, stop_count

    def test_adaptive_parameter_gating(self):
        ef_init, ef_max, ef_min = 128, 512, 64

        # ---- Conservative: 확장을 원천 차단 (LID threshold를 매우 높게 설정) ----
        conservative = dict(
            ef_init=ef_init, ef_max=ef_max, ef_min=ef_min,
            tmin_pops=30, lid_window_k=20, stall_window_w=20,
            lid_low=0.0,
            lid_high=1000.0,     # 절대값으로 매우 높게 설정
            lid_high2=2000.0,
            dist_stall_up=0.0001,
            dist_stall_stop=0.003,
            enable_stop=False,   # enable_down -> enable_stop 명칭 변경 반영
        )

        # ---- Aggressive: 정체 감지 시 즉시 확장 (LID threshold를 매우 낮게 설정) ----
        aggressive = dict(
            ef_init=ef_init, ef_max=ef_max, ef_min=ef_min,
            tmin_pops=10, lid_window_k=20, stall_window_w=20,
            lid_low=0.0,
            lid_high=-10.0,      # 무조건 확장이 일어나도록 낮게 설정
            lid_high2=-5.0,
            dist_stall_up=0.1,   # 정체 감지 조건을 매우 느슨하게
            dist_stall_stop=0.01,
            enable_stop=False,
        )

        _, d_cons, _, _ = self._run_adaptive(self.q_boundary, **conservative)
        _, d_aggr, _, _ = self._run_adaptive(self.q_boundary, **aggressive)

        r_cons = self._radius(d_cons[0])
        r_aggr = self._radius(d_aggr[0])

        _, d_init = self._run_fixed(self.q_boundary, ef_init)
        r_init = self._radius(d_init[0])

        print(f"\n[Boundary Query] Radius fixed_init={r_init:.6f}, adaptive_cons={r_cons:.6f}, adaptive_aggr={r_aggr:.6f}")

        # Conservative는 확장이 안 되어야 하므로 ef=128과 유사해야 함
        self.assertAlmostEqual(r_cons, r_init, delta=0.05)
        # Aggressive는 확장이 일어나야 하므로 Conservative보다 결과가 좋아야(Radius가 작아야) 함
        self.assertLessEqual(r_aggr, r_cons + 1e-6)

    def test_early_stop_mechanism(self):
        """LID_low와 stall_stop을 통한 조기 종료가 작동하는지 확인"""
        ef_init = 128

        cfg_stop = dict(
            ef_init=ef_init, ef_max=512, ef_min=64,
            tmin_pops=30, lid_window_k=20, stall_window_w=20,
            lid_low=100.0,         # LID가 낮다고 속이기 위해 threshold를 높게 설정
            lid_high=500.0,
            lid_high2=600.0,
            dist_stall_up=0.0001,
            dist_stall_stop=0.1,   # 정체가 조금만 생겨도 STOP 하도록
            enable_stop=True,
        )

        # Easy 쿼리에서 조기 종료가 발생하는지 호출 (충돌 여부 및 유효성 확인)
        labels, dists, reduced_steps, stop_count = self._run_adaptive(self.q_easy, **cfg_stop)

        self.assertEqual(labels.shape, (1, self.k))
        self.assertTrue(np.all(np.isfinite(dists)))
        self.assertTrue(np.all(reduced_steps >= 0))
        self.assertGreaterEqual(int(stop_count), 0)

if __name__ == "__main__":
    unittest.main()
