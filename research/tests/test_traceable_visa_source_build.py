"""Synthetic utility tests only; no training/encoder/source scientific results."""
import copy
import inspect
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments import source_build_traceable_visa as job


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cesad_source_gate_test_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.asset = self.base/'unit_test_input.txt'
        self.asset.write_text('unit test only',encoding='ascii')
        self.manifest = dict(status='IMPLEMENTATION_SEALED_AWAITING_TRAINING_REVIEW',
            config_sha256='unit-test-config',locked_files=[job.audit.ident(self.asset)])
        self.mp = self.base/'SOURCE_INPUT_MANIFEST.json'
        job.audit.js(self.mp,self.manifest)
        self.approval = dict(decision='APPROVED_SOURCE_BUILD_ONLY',reviewer='UNIT_TEST_NOT_REAL_APPROVAL',
            source_input_manifest_sha256=job.audit.sha(self.mp),config_sha256='unit-test-config')

    def save_approval(self):
        job.audit.js(self.base/'SOURCE_BUILD_APPROVAL.json',self.approval)

    def test_missing_approval(self):
        with self.assertRaises(FileNotFoundError):
            job.approval_gate(self.base)

    def test_missing_manifest(self):
        self.mp.unlink()
        with self.assertRaises(FileNotFoundError):
            job.approval_gate(self.base)

    def test_feature_approval_is_not_training_approval(self):
        self.approval['decision'] = 'NEW_SOURCE_FEATURE_ARTIFACT_REVIEW_PASSED_ONLY'
        self.save_approval()
        with self.assertRaisesRegex(RuntimeError,'Separate explicit'):
            job.approval_gate(self.base)

    def test_reviewer_required(self):
        self.approval['reviewer'] = ''
        self.save_approval()
        with self.assertRaisesRegex(RuntimeError,'Separate explicit'):
            job.approval_gate(self.base)

    def test_manifest_hash_required(self):
        self.approval['source_input_manifest_sha256'] = 'wrong'
        self.save_approval()
        with self.assertRaisesRegex(RuntimeError,'manifest hash'):
            job.approval_gate(self.base)

    def test_config_hash_required(self):
        self.approval['config_sha256'] = 'wrong'
        self.save_approval()
        with self.assertRaisesRegex(RuntimeError,'config hash'):
            job.approval_gate(self.base)

    def test_changed_input_rejected(self):
        self.save_approval()
        self.asset.write_text('changed',encoding='ascii')
        with self.assertRaisesRegex(RuntimeError,'Locked input changed'):
            job.approval_gate(self.base)

    def test_unknown_manifest_status_rejected(self):
        self.manifest['status'] = 'LEGACY_BLOCKED'
        job.audit.js(self.mp,self.manifest)
        self.approval['source_input_manifest_sha256'] = job.audit.sha(self.mp)
        self.save_approval()
        with self.assertRaisesRegex(RuntimeError,'Unrecognized'):
            job.approval_gate(self.base)

    def test_valid_guard_does_not_create_training_output(self):
        self.save_approval()
        manifest,_,_ = job.approval_gate(self.base)
        self.assertEqual(manifest,self.manifest)
        self.assertEqual({p.name for p in self.base.iterdir()},
            {'unit_test_input.txt','SOURCE_INPUT_MANIFEST.json','SOURCE_BUILD_APPROVAL.json'})

    def test_runner_stops_before_training_imports(self):
        original_gate = job.approval_gate
        with patch.object(job,'approval_gate',side_effect=lambda: original_gate(self.base)):
            with patch.object(job,'refs',side_effect=AssertionError('must never be called')) as imports:
                with self.assertRaises(FileNotFoundError):
                    job.source_build()
                imports.assert_not_called()

    def test_old_cache_path_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Non-certified'):
            job.fresh_only({'patch_features':{'path':str(self.base/'old.npy')}})

    def test_wrong_dtype_shape_rejected(self):
        with patch.object(job.audit,'WORK',self.base):
            target = self.base/'certified_source/source_patch_float32.npy'
            target.parent.mkdir()
            np.save(target,np.ones((2,2),np.float16))
            with self.assertRaisesRegex(RuntimeError,'dtype/shape'):
                job.fresh_only({'patch_features':{'path':str(target)}})


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.images = [dict(category=name) for name in ['b','a','b','a']]
        self.labels = np.asarray([[0,1,0,1],[1,0,1,0],[0,0,1,1],[1,1,0,0]],np.uint8)

    def test_auxiliary_plan_preserves_labels_and_is_deterministic(self):
        a = job.aux_pool_plan(self.images,self.labels)
        b = job.aux_pool_plan(self.images,self.labels)
        for left,right in zip(a,b):
            np.testing.assert_array_equal(left,right)
        pairs,y,cat = a
        for i,(pos,pid) in enumerate(pairs):
            self.assertEqual(y[i],self.labels[pos,pid])
            self.assertEqual(cat[i],self.images[pos]['category'])
        self.assertEqual(list(cat),sorted(cat.tolist()))

    def test_ces_origin_and_true_labels(self):
        images = [dict(category='a')]*3
        labels = np.asarray([[0,0,0,0],[0,1,0,1],[1,1,0,0]],np.uint8)
        pairs,y,_,origins = job.ces_pool_plan(images,labels)
        for (pos,pid),label,origin in zip(pairs,y,origins):
            self.assertEqual(label,labels[pos,pid])
            if origin == 'normal_image':
                self.assertEqual(pos,0)
            elif origin == 'anomaly_mask':
                self.assertEqual(label,1)
            else:
                self.assertEqual(label,0)

    def test_stage_I_has_twenty_balanced_updates(self):
        _,y,cat = job.aux_pool_plan(self.images,self.labels)
        batches = job.state_batches(y,cat)
        self.assertEqual(len(batches),20)
        for ids in batches:
            self.assertEqual(np.count_nonzero(y[ids] == 0),np.count_nonzero(y[ids] == 1))
            self.assertEqual(len(ids),len(set(ids.tolist())))

    def test_adapter_plan_epochs_and_permutations(self):
        b = job.adapter_batches(9000)
        self.assertEqual(len(b['epochs']),20)
        for epoch in range(10):
            ids = b['flat'][b['offsets'][epoch*2]:b['offsets'][epoch*2+2]]
            np.testing.assert_array_equal(np.sort(ids),np.arange(9000))
        self.assertEqual(b['offsets'][1]-b['offsets'][0],8192)

    def test_prepare_has_no_fit_or_model_construction(self):
        source = inspect.getsource(job.prepare)
        for forbidden in ('np.linalg.svd(', 'torch.optim.', '.backward(', 'run_stage_I(', 'run_adapter(', 'run_ces('):
            self.assertNotIn(forbidden,source)


if __name__ == '__main__':
    unittest.main()
