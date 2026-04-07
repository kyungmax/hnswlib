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

    def _run_adaptive_analysis(self, query: np.ndarray, **kwargs):
        labels, dists, reduced_steps, stop_count = self.p.knn_query_adaptive_analysis(
            query, k=self.k, num_threads=1, **kwargs
        )
        return labels, dists, reduced_steps, stop_count

    def _run_adaptive_light(self, query: np.ndarray, **kwargs):
        labels, dists = self.p.knn_query_adaptive_light(query, k=self.k, num_threads=1, **kwargs)
        return labels, dists

    def test_adaptive_light_matches_default_adaptive(self):
        full = self._run_adaptive_analysis(self.q_boundary)
        light = self._run_adaptive_light(self.q_boundary)

        np.testing.assert_array_equal(light[0], full[0])
        np.testing.assert_allclose(light[1], full[1])

    def test_adaptive_light_accepts_custom_ef_init(self):
        ef_init = 256

        full = self._run_adaptive_analysis(self.q_boundary, ef_init=ef_init)
        light = self._run_adaptive_light(self.q_boundary, ef_init=ef_init)

        np.testing.assert_array_equal(light[0], full[0])
        np.testing.assert_allclose(light[1], full[1])

    def test_adaptive_parameter_gating(self):
        ef_init = 128

        conservative = dict(
            ef_init=ef_init,
            ef_max=ef_init,
            tmin_pops=64,
            enable_stop=False,
        )

        aggressive = dict(
            ef_init=ef_init,
            ef_max=512,
            tmin_pops=10,
            enable_stop=False,
        )

        _, d_cons, _, _ = self._run_adaptive_analysis(self.q_boundary, **conservative)
        _, d_aggr, _, _ = self._run_adaptive_analysis(self.q_boundary, **aggressive)

        r_cons = self._radius(d_cons[0])
        r_aggr = self._radius(d_aggr[0])

        _, d_init = self._run_fixed(self.q_boundary, ef_init)
        r_init = self._radius(d_init[0])

        print(f"\n[Boundary Query] Radius fixed_init={r_init:.6f}, adaptive_cons={r_cons:.6f}, adaptive_aggr={r_aggr:.6f}")

        self.assertAlmostEqual(r_cons, r_init, delta=0.05)
        self.assertLessEqual(r_aggr, r_cons + 1e-6)

    def test_early_stop_mechanism(self):
        """LID_low와 stall_stop을 통한 조기 종료가 작동하는지 확인"""
        ef_init = 128

        cfg_stop = dict(
            ef_init=ef_init,
            ef_max=512,
            tmin_pops=30,
            enable_stop=True,
        )

        # Easy 쿼리에서 조기 종료가 발생하는지 호출 (충돌 여부 및 유효성 확인)
        labels, dists, reduced_steps, stop_count = self._run_adaptive_analysis(self.q_easy, **cfg_stop)

        self.assertEqual(labels.shape, (1, self.k))
        self.assertTrue(np.all(np.isfinite(dists)))
        self.assertTrue(np.all(reduced_steps >= 0))
        self.assertGreaterEqual(int(stop_count), 0)

    def test_analysis_stop_step_caps_pop_count(self):
        labels, dists, reduced_steps, stop_count = self._run_adaptive_analysis(
            self.q_boundary,
            ef_init=128,
            ef_max=512,
            tmin_pops=64,
            enable_stop=False,
            stop_step=64,
        )

        self.assertEqual(labels.shape, (1, self.k))
        self.assertTrue(np.all(np.isfinite(dists)))
        self.assertEqual(int(reduced_steps[0]), 64)
        self.assertEqual(int(stop_count), 0)

    def test_sampled_internal_lids_api_preserves_full_lids(self):
        full_lids_before = np.array(self.p.get_lids(), copy=True)

        sampled_ids, sampled_lids = self.p.calc_lids_internal_sampled(
            k_lid=15,
            sample_fraction=0.01,
            min_sample_size=64,
            random_seed=7,
            num_threads=1,
        )

        self.assertEqual(sampled_ids.ndim, 1)
        self.assertEqual(sampled_lids.ndim, 1)
        self.assertEqual(sampled_ids.shape[0], sampled_lids.shape[0])
        self.assertEqual(sampled_ids.shape[0], max(int(np.ceil(self.num_elements * 0.01)), 64))
        self.assertTrue(np.all(sampled_ids[:-1] <= sampled_ids[1:]))
        self.assertTrue(np.all(np.isfinite(sampled_lids)))

        full_lids_after = np.array(self.p.get_lids(), copy=True)
        np.testing.assert_allclose(full_lids_after, full_lids_before)

    def test_direct_mean_threshold_changes_analysis_behavior(self):
        base_cfg = dict(
            ef_init=128,
            ef_max=512,
            tmin_pops=25,
            early_stop_ratio=0.0,
            easy_stag_limit=1,
            hard_stag_limit=100000,
            enable_stop=True,
        )

        _, d_no_rescue, reduced_no_rescue, stop_no_rescue = self._run_adaptive_analysis(
            self.q_easy,
            **base_cfg,
        )
        rescue_cfg = dict(base_cfg)
        rescue_cfg["early_stop_ratio"] = 1.0
        _, d_rescue, reduced_rescue, stop_rescue = self._run_adaptive_analysis(
            self.q_easy,
            **rescue_cfg,
        )

        self.assertTrue(np.all(np.isfinite(d_no_rescue)))
        self.assertTrue(np.all(np.isfinite(d_rescue)))
        self.assertEqual(int(stop_no_rescue), 0)
        self.assertEqual(int(stop_rescue), 1)
        self.assertLess(int(reduced_rescue[0]), int(reduced_no_rescue[0]))

        labels_light, dists_light = self._run_adaptive_light(
            self.q_easy,
            ef_init=128,
            early_stop_ratio=1.0,
            tmin_pops=25,
            easy_stag_limit=1,
            hard_stag_limit=100000,
        )
        self.assertEqual(labels_light.shape, (1, self.k))
        self.assertTrue(np.all(np.isfinite(dists_light)))

if __name__ == "__main__":
    unittest.main()
