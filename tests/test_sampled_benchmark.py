from __future__ import annotations

import csv
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_sampled_videos import (
    ALL_DIMENSIONS,
    DEFAULT_DATASET,
    DEFAULT_MODEL,
    build_sampled_dataset,
    load_model_rows,
    sample_rows,
    write_reports,
)
from metric.timing import EvaluationTimer
import runner
from runtime_config import api_model_name, configure_runtime, droid_weights_path, fastvqa_weights_path, keye_model_path, vfimamba_weights_path


EXPECTED_SEED_42 = [2, 11, 25, 34, 44, 53, 62, 79, 82, 100, 107, 111, 121, 132, 144, 154]


class SamplingTests(unittest.TestCase):
    def test_both_entries_use_same_paths_for_migrated_dataset(self):
        info, rows = load_model_rows(DEFAULT_DATASET, DEFAULT_MODEL)
        direct = [p for item in runner.read_dataset_json(str(DEFAULT_DATASET)) if item['model'] == DEFAULT_MODEL for p in item['video_list']]
        self.assertEqual(set(direct), {item['video'] for item in rows})
        chosen, _ = sample_rows(rows, 42)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'FOURDWORLDBENCH_VIDEO_ROOT': str(DEFAULT_DATASET.parent.parent)}):
            manifest = Path(tmp) / 'samples.json'
            manifest.write_text(json.dumps(build_sampled_dataset(info, chosen, DEFAULT_MODEL)), encoding='utf-8')
            nested = [p for item in runner.read_dataset_json(str(manifest)) for p in item['video_list']]
            self.assertEqual(nested, [item['video'] for item in chosen])

    def test_cli_selects_all_local_weight_paths_and_api_model(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            configure_runtime(SimpleNamespace(video_root=None, weights_root=root, api_model='alternate-model'))
            self.assertEqual(keye_model_path(), root / 'checkpoints/Keye-VL-1_5-8B')
            self.assertEqual(droid_weights_path(), root / 'checkpoints/droid.pth')
            self.assertEqual(vfimamba_weights_path('VFIMamba'), root / 'checkpoints/VFIMamba/model.pkl')
            self.assertEqual(fastvqa_weights_path('/old/fastvqa.pth'), root / 'pretrained_weights/fastvqa.pth')
            self.assertEqual(api_model_name(), 'alternate-model')

    def test_runner_resolves_prefixed_relative_paths_in_legacy_manifests(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            dataset_root = Path(tmp) / 'legacy' / 'video_set'
            dataset_root.mkdir(parents=True)
            video = dataset_root / '0001.mp4'
            video.touch()
            manifest = dataset_root / 'manifest.json'
            manifest.write_text(json.dumps({
                'dataset_info': {'base_path': 'legacy/video_set'},
                'models': [{'model_name': 'test', 'conditions': [{'prompts': [
                    {'generated_videos': ['legacy/video_set/0001.mp4']},
                ]}]}],
            }), encoding='utf-8')
            self.assertEqual(runner.read_dataset_json(str(manifest))[0]['video_list'], [str(video)])

    def test_seed_42_selection_is_stable(self):
        _, rows = load_model_rows(DEFAULT_DATASET, DEFAULT_MODEL)
        selected, intervals = sample_rows(rows, 42)
        self.assertEqual([item["video_id"] for item in selected], EXPECTED_SEED_42)
        self.assertEqual(len(intervals), 16)
        self.assertTrue(all(len(item["candidate_video_ids"]) == 10 for item in intervals))

    def test_full_report_has_208_video_rows_and_13_summary_rows(self):
        _, rows = load_model_rows(DEFAULT_DATASET, DEFAULT_MODEL)
        selected, _ = sample_rows(rows, 42)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_reports(output, ALL_DIMENSIONS, selected, {})
            with (output / "video_dimension_timings.csv").open(encoding="utf-8") as handle:
                video_rows = list(csv.DictReader(handle))
            with (output / "dimension_timing_summary.csv").open(encoding="utf-8") as handle:
                dimension_rows = list(csv.DictReader(handle))
            self.assertEqual(len(video_rows), 16 * 13)
            self.assertEqual(len(dimension_rows), 13)


class TimingTests(unittest.TestCase):
    def test_success_failure_and_average_are_recorded(self):
        timer = EvaluationTimer("fake_dimension", "cpu")
        load_started = timer.start_model_load()
        time.sleep(0.001)
        timer.finish_model_load(load_started)
        with timer.measure_video("ok.mp4"):
            time.sleep(0.001)
        failed = timer.start_video("failed.mp4")
        time.sleep(0.001)
        timer.finish_video(failed, RuntimeError("boom"))
        result = timer.to_dict()
        self.assertGreater(result["model_load_sec"], 0)
        self.assertEqual(result["successful_videos"], 1)
        self.assertEqual(result["failed_videos"], 1)
        self.assertAlmostEqual(
            result["average_video_compute_sec"],
            result["videos"][0]["compute_sec"],
            places=5,
        )
        self.assertEqual(result["videos"][1]["status"], "failed")
        self.assertIn("boom", result["videos"][1]["error"])

    def test_runner_passes_one_timer_through_model_load_and_video_work(self):
        timer = EvaluationTimer("fake", "cpu")

        def compute_fake(json_dir, device, submodules, **kwargs):
            received = submodules["evaluation_timer"]
            with received.measure_model_load():
                time.sleep(0.001)
            details = []
            for path in ("one.mp4", "two.mp4"):
                with received.measure_video(path):
                    time.sleep(0.001)
                details.append({"video_path": path, "score": 1.0})
            return 1.0, details

        with patch.object(
            runner,
            "dynamic_import_dimension",
            return_value=SimpleNamespace(compute_fake=compute_fake),
        ):
            result = runner.run_dimension(
                "fake", [], "unused.json", device="cpu", timer=timer
            )
        timing = timer.to_dict()
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(timing["successful_videos"], 2)
        self.assertGreater(timing["model_load_sec"], 0)


if __name__ == "__main__":
    unittest.main()
