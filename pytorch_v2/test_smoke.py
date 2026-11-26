import unittest
import torch
from model import Model, Config, XIELU, encode_cds, encode_protein, ALLOWED, AA_ID
from uncertainty import summarize, predict, fit_temperature
from train import collate
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
    def test_temperature(self):
        x=torch.tensor([[4.,0.,float("-inf")],[4.,0.,float("-inf")]])
        # Class IDs 0/2 are special in real vocabulary, so use real-shaped IDs.
        x=torch.tensor([[float("-inf"),float("-inf"),float("-inf"),4.,0.],
                        [float("-inf"),float("-inf"),float("-inf"),4.,0.]])
        t=fit_temperature(x,torch.tensor([3,4]))
        self.assertGreater(t,1.)

if __name__=="__main__":
    unittest.main()
