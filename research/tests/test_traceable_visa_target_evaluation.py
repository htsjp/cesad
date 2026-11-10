"""Utility/gate tests only: no model or real target metrics are executed."""
import inspect
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments import evaluate_traceable_visa_targets as job


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cesad_target_guard_')
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.asset=self.base/'unit_test_input.txt';self.asset.write_text('unit test only')
        self.config=dict(endpoint_order=list(job.ENDPOINTS),endpoint_roles=job.ROLES,
                         main_model='update30',selection='NONE_REPORT_BOTH')
        self.cp=self.base/'TARGET_EVALUATION_CONFIG.json';job.audit.js(self.cp,self.config)
        self.manifest=dict(status='PREPARED_TARGET_EVALUATION_LOCKED',config_sha256=job.audit.sha(self.cp),
                           source_seal_sha256='unit-test-source',locked_files=[job.audit.ident(self.asset),job.audit.ident(self.cp)])
        self.mp=self.base/'TARGET_EVALUATION_INPUT_MANIFEST.json';job.audit.js(self.mp,self.manifest)
        self.approval=dict(decision='APPROVED_TARGET_EVALUATION_ONLY',reviewer='UNIT_TEST_NOT_REAL_APPROVAL',
            input_manifest_sha256=job.audit.sha(self.mp),config_sha256=self.manifest['config_sha256'],
            source_seal_sha256=self.manifest['source_seal_sha256'])

    def approve(self):
        job.audit.js(self.base/'TARGET_EVALUATION_APPROVAL.json',self.approval)

    def test_missing_approval(self):
        with self.assertRaises(FileNotFoundError):job.gate(self.base)

    def test_source_approval_is_insufficient(self):
        self.approval['decision']='APPROVED_SOURCE_BUILD_ONLY';self.approve()
        with self.assertRaisesRegex(RuntimeError,'Separate explicit'):job.gate(self.base)

    def test_reviewer_required(self):
        self.approval['reviewer']='';self.approve()
        with self.assertRaisesRegex(RuntimeError,'Separate explicit'):job.gate(self.base)

    def test_manifest_hash_required(self):
        self.approval['input_manifest_sha256']='wrong';self.approve()
        with self.assertRaisesRegex(RuntimeError,'input-manifest hash'):job.gate(self.base)

    def test_config_hash_required(self):
        self.approval['config_sha256']='wrong';self.approve()
        with self.assertRaisesRegex(RuntimeError,'config hash'):job.gate(self.base)

    def test_source_seal_hash_required(self):
        self.approval['source_seal_sha256']='wrong';self.approve()
        with self.assertRaisesRegex(RuntimeError,'source seal'):job.gate(self.base)

    def test_modified_input_rejected(self):
        self.approve();self.asset.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Locked input changed'):job.gate(self.base)

    def test_valid_guard_only_does_not_evaluate(self):
        self.approve();_,config,_,_=job.gate(self.base)
        self.assertEqual(config['main_model'],'update30')

    def test_no_encoder_before_approval(self):
        original=job.gate
        with patch.object(job,'gate',side_effect=lambda:original(self.base)):
            with patch.object(job,'Encoder',side_effect=AssertionError('no encoder')) as encoder:
                with self.assertRaises(FileNotFoundError):job.run()
                encoder.assert_not_called()


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(endpoint_order=list(job.ENDPOINTS),endpoint_roles=dict(job.ROLES),
                         main_model='update30',selection='NONE_REPORT_BOTH')

    def test_fixed_roles(self):
        job.validate_roles(self.config)

    def test_reversed_endpoint_order_rejected(self):
        self.config['endpoint_order']=['update30','initialization']
        with self.assertRaises(RuntimeError):job.validate_roles(self.config)

    def test_diagnostic_cannot_be_main(self):
        self.config['main_model']='initialization'
        with self.assertRaises(RuntimeError):job.validate_roles(self.config)

    def test_target_selection_disallowed(self):
        self.config['selection']='BEST_TARGET_AP'
        with self.assertRaises(RuntimeError):job.validate_roles(self.config)

    def test_odd_image_quadrants_cover_without_overlap(self):
        occupancy=np.zeros((13,11),np.uint8)
        for x1,y1,x2,y2 in job.crop_boxes(11,13):occupancy[y1:y2,x1:x2]+=1
        np.testing.assert_array_equal(occupancy,np.ones_like(occupancy))
        self.assertEqual(job.crop_boxes(11,13),[(0,0,5,6),(5,0,11,6),(0,6,5,13),(5,6,11,13)])

    def test_fusion_keeps_endpoints_and_quadrants(self):
        q=torch.zeros(2,5,37,37)
        for k in range(2):
            q[k,0]=2+10*k
            for j in range(1,5):q[k,j]=j+10*k
        actual=job.fuse_views(q).numpy()
        for k in range(2):
            for j,(x,y) in enumerate([(0,0),(37,0),(0,37),(37,37)],1):
                np.testing.assert_array_equal(actual[k,y:y+37,x:x+37],np.full((37,37),.5*(2+10*k)+.5*(j+10*k)))

    def test_wrong_view_shape_rejected(self):
        with self.assertRaises(RuntimeError):job.fuse_views(torch.zeros(2,4,37,37))

    def test_prepare_never_calls_forward_or_metrics(self):
        text=inspect.getsource(job.prepare)
        for forbidden in ('Encoder(', 'Scorer(', 'weighted_pixel_metrics(', 'aupro(', 'predict(', 'evaluate('):
            self.assertNotIn(forbidden,text)

    def test_both_endpoints_share_single_encoding(self):
        text=inspect.getsource(job.predict)
        self.assertEqual(text.count('encoder.extract('),1)
        self.assertNotIn('for name in ENDPOINTS',text)

    def test_optional_annotation_column_only(self):
        self.assertEqual(job.identity_value({},'annotation_note'),'')
        self.assertNotEqual(job.identity_value({},'annotation_note'),job.identity_value({'annotation_note':'EMPTY_MASK'},'annotation_note'))
        with self.assertRaises(KeyError):job.identity_value({},'gt_sha256')


if __name__ == '__main__':unittest.main()
