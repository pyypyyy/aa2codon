import unittest
import torch
from model import Model, Config, XIELU, encode_cds, encode_protein, ALLOWED, AA_ID
from uncertainty import summarize, predict, fit_temperature
from train import collate, BucketSampler, loader, make_optimizer, make_scaler, epoch
from torch.nn import functional as F

class SmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.m=Model(Config(d_model=16,heads=2,layers=1,d_ff=32,dropout=0,max_length=32))
    def test_xielu(self):
        f=XIELU()
        x=torch.tensor([-100.,-1.,0.,1.,100.],requires_grad=True)
        y=f(x); y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(float(x.grad[2]),.5,places=5)
    def test_train_and_predict(self):
        rows=[encode_cds("ATGTTTCTGTAA")[:2],encode_cds("ATGGCT")[:2]]
        a,p,y=collate(rows)
        logits,_=self.m(a,p)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in self.m.parameters() if v.grad is not None))
        r=predict([self.m,self.m],"MFL*")
        self.assertEqual(encode_cds(r["dna"])[2],"MFL*")
        self.assertTrue(torch.allclose(r["total"],r["aleatoric"]+r["epistemic"],atol=1e-6))
        self.assertTrue(torch.allclose(r["epistemic"],torch.zeros(4),atol=1e-6))
    def test_uncertainty(self):
        allowed=torch.tensor([[True,True]])
        r=summarize(torch.tensor([[[1.,0.]],[[0.,1.]]]),allowed)
        self.assertAlmostEqual(float(r["epistemic"]),1.,places=6)
        self.assertAlmostEqual(float(r["aleatoric"]),0.,places=6)
        r=summarize(torch.full((2,1,2),.5),allowed)
        self.assertAlmostEqual(float(r["aleatoric"]),1.,places=6)
        self.assertAlmostEqual(float(r["epistemic"]),0.,places=6)
    def test_causal_prefix_and_attention_parity(self):
        self.m.eval()
        a,p,y=collate([encode_cds("ATGTTTCTGTAA")[:2]])
        with torch.no_grad():
            z,w=self.m(a,p,attention=True)
            plain,_=self.m(a,p)
            self.assertTrue(torch.allclose(z,plain,atol=2e-5,rtol=2e-5))
            self.assertEqual(tuple(w[0].shape),(1,2,p.shape[1],a.shape[1]))
            self.assertTrue(torch.allclose(w[0].sum(-1),torch.ones_like(w[0].sum(-1)),atol=1e-6))
            changed=p.clone()
            changed[0,-1]=1
            other,_=self.m(a,changed)
            self.assertTrue(torch.allclose(plain[:,:-1],other[:,:-1],atol=2e-5,rtol=2e-5))
    def test_padding_and_optimizer_step(self):
        rows=[encode_cds("ATGTTTCTGTAA")[:2],encode_cds("ATGGCT")[:2]]
        a,p,y=collate(rows)
        self.m.eval()
        with torch.no_grad():
            z,w=self.m(a,p,attention=True)
            self.assertTrue(torch.isfinite(z.gather(-1,y.unsqueeze(-1))).all())
            self.assertTrue((w[0][1,...,a[1].eq(0)]==0).all())
            self.assertTrue(torch.allclose(z,self.m(a,p)[0],atol=2e-5,rtol=2e-5))
        opt=make_optimizer(self.m,3e-4,.01)
        result=epoch(self.m,loader(rows,2,True,7),torch.device("cpu"),opt,scaler=make_scaler(torch.device("cpu")))
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
    def test_bucket_coverage(self):
        rows=[([0]*i,[0]*i) for i in range(1,72)]
        sampler=BucketSampler(rows,8,7)
        batches=list(sampler)
        self.assertEqual(len(batches),len(sampler))
        self.assertEqual(sorted(i for batch in batches for i in batch),list(range(len(rows))))
        self.assertTrue(all(len(batch)<=8 for batch in batches))
    def test_temperature(self):
        x=torch.tensor([[4.,0.,float("-inf")],[4.,0.,float("-inf")]])
        # Class IDs 0/2 are special in real vocabulary, so use real-shaped IDs.
        x=torch.tensor([[float("-inf"),float("-inf"),float("-inf"),4.,0.],
                        [float("-inf"),float("-inf"),float("-inf"),4.,0.]])
        t=fit_temperature(x,torch.tensor([3,4]))
        self.assertGreater(t,1.)

if __name__=="__main__":
    unittest.main()
