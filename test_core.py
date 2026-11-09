"""Synthetic mathematical smoke tests only; not experimental evidence."""
import unittest
import numpy as np
import torch
from cesad_core import CESHead, LinearHead, NormalizedLinearHead, ntr_loss_from_logits

class CoreTests(unittest.TestCase):
    def test_matched_logits_and_frozen_geometry(self):
        rng=np.random.default_rng(0)
        sn=rng.normal(size=1024).astype(np.float32);sn/=np.linalg.norm(sn)
        sa=rng.normal(size=1024).astype(np.float32);sa/=np.linalg.norm(sa)
        m=(sa+sn)/2;d=(sa-sn)/2;a=float(np.linalg.norm(d));v=d/a
        head=CESHead(m,a,v)
        z=torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(16,1024)).astype(np.float32)),dim=-1)
        w=torch.nn.functional.normalize(head.m+head.alpha*torch.nn.functional.normalize(head.v,dim=0),dim=0)-torch.nn.functional.normalize(head.m-head.alpha*torch.nn.functional.normalize(head.v,dim=0),dim=0)
        linear=LinearHead(w.detach().numpy());normalized=NormalizedLinearHead(w.detach().numpy(),float(w.norm()))
        self.assertLess(float((head(z)-linear(z)).abs().max().detach()),1e-6)
        self.assertLess(float((head(z)-normalized(z)).abs().max().detach()),1e-6)
        self.assertEqual(sum(p.numel() for p in head.parameters()),1024)
        self.assertFalse(head.m.requires_grad);self.assertFalse(head.alpha.requires_grad)
    def test_ntr_real_top10_percent(self):
        scores=torch.arange(20,dtype=torch.float32,requires_grad=True)
        labels=torch.zeros(20)
        loss=ntr_loss_from_logits(scores,labels,.1)
        self.assertTrue(torch.equal(loss,torch.nn.functional.softplus(scores[-2:]).mean()))
        loss.backward();self.assertEqual(int((scores.grad!=0).sum()),2)

if __name__=='__main__':unittest.main()
