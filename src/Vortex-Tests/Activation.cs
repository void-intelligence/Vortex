// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Activation;

namespace VortexTests
{
    [TestClass]
    public class VortexActivation
    {
        [TestMethod]
        public void ArctanTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Arctan().Forward(a);
            b.InMap(Math.Atan);
            Assert.IsTrue(a == b, "Arctan Activation successful");
        }

        [TestMethod]
        public void ArctanPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Arctan().Backward(a);
            b.InMap((x) => 1 / (1 + Math.Pow(x, 2)));
            Assert.IsTrue(a == b, "Arctan Derivative successful");
        }

        [TestMethod]
        public void BinaryStepTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new BinaryStep().Forward(a);
            b.InMap((x) => (x < 0) ? 0 : 1);
            Assert.IsTrue(a == b, "BinaryStep Activation successful");
        }

        [TestMethod]
        public void BinaryStepPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new BinaryStep().Backward(a);
            b.InMap((x) => 0);
            Assert.IsTrue(a == b, "BinaryStep Derivative successful");
        }

        [TestMethod]
        public void BipolarSigmoidTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new BipolarSigmoid().Forward(a);
            b.InMap((x) => -1 + 2 / (1 + Math.Exp(-x)));
            Assert.IsTrue(a == b, "Bipolar Sigmoid Activation successful");
        }

        [TestMethod]
        public void BipolarSigmoidPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new BipolarSigmoid().Backward(a);
            b.InMap((x) => 0.5 * (1 + (-1 + 2 / (1 + Math.Exp(-x)))) * (1 - (-1 + 2 / (1 + Math.Exp(-x)))));
            Assert.IsTrue(a == b, "Bipolar Sigmoid Derivative successful");
        }

        [TestMethod]
        public void ELUTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            double Alpha = 0.01;
            a = new ELU(new ELUSettings()).Forward(a);
            b.InMap((x) => (x >= 0) ? x : Alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(a == b, "ELU Activation successful");
        }

        [TestMethod]
        public void ELUPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            double Alpha = 0.01;
            a = new ELU(new ELUSettings()).Backward(a);
            b.InMap((x) => (x >= 0) ? 1 : Alpha * Math.Exp(x));
            Assert.IsTrue(a == b, "ELU Derivative successful");
        }

        [TestMethod]
        public void HardSigmoidTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new HardSigmoid().Forward(a);
            b.InMap((x) => (x < 0) ? 0 : (x < 1) ? x : 1);
            Assert.IsTrue(a == b, "Hard Sigmoid Activation successful");
        }

        [TestMethod]
        public void HardSigmoidPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new HardSigmoid().Backward(a);
            b.InMap((x) => (x > 1 || x < 0) ? 0 : 1);
            Assert.IsTrue(a == b, "Hard Sigmoid Derivative successful");
        }

        [TestMethod]
        public void HardTanhTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new HardTanh().Forward(a);
            b.InMap((x) => (x < -1) ? -1 : (x > 1) ? 1 : x);
            Assert.IsTrue(a == b, "Hard Tanh Activation successful");
        }

        [TestMethod]
        public void HardTanhPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new HardTanh().Backward(a);
            b.InMap((x) => (x < -1) ? 0 : (x > 1) ? 0 : 1);
            Assert.IsTrue(a == b, "Hard Tanh Derivative successful");
        }

        [TestMethod]
        public void IdentityTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Identity().Forward(a);
            b.InMap((x) => x);
            Assert.IsTrue(a == b, "Identity Activation successful");
        }

        [TestMethod]
        public void IdentityPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Identity().Backward(a);
            b.InMap((x) => 1);
            Assert.IsTrue(a == b, "Identity Derivative successful");
        }

        [TestMethod]
        public void LogitTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Logit().Forward(a);
            b.InMap((x) => Math.Log(x / (1 - x)));
            Assert.IsTrue(a == b, "Logit Activation successful");
        }

        [TestMethod]
        public void LogitPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Logit().Backward(a);
            b.InMap((x) => (-1 / Math.Pow(x, 2)) - (1 / (Math.Pow((1 - x), 2))));
            Assert.IsTrue(a == b, "Logit Derivative successful");
        }

        [TestMethod]
        public void LReLUTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            double Alpha = 0.01;
            a = new LReLU(new LReLUSettings()).Forward(a);
            b.InMap((x) => Math.Max(x, Alpha));
            Assert.IsTrue(a == b, "ReLU Activation successful");
        }

        [TestMethod]
        public void LReLUPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            double Alpha = 0.01;
            a = new LReLU(new LReLUSettings()).Backward(a);
            b.InMap((x) => (x > 0) ? 1 : Alpha);
            Assert.IsTrue(a == b, "ReLU Derivative successful");
        }

        [TestMethod]
        public void MishTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Mish().Forward(a);
            b.InMap((x) => x * Math.Tanh(Math.Log(1 + Math.Exp(x))));
            Assert.IsTrue(a == b, "Mish Activation successful");
        }

        [TestMethod]
        public void MishPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Mish().Backward(a);
            b.InMap((x) => ((Math.Exp(x) * (4 * (x + 1) + 4 * Math.Exp(2 * x) + Math.Exp(3 * x) + Math.Exp(x) * (4 * x + 6))) / Math.Pow((2 * Math.Exp(x) + Math.Exp(2 * x) + 2), 2)));
            Assert.IsTrue(a == b, "Mish Derivative successful");
        }

        [TestMethod]
        public void ReLUTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new ReLU().Forward(a);
            b.InMap((x) => Math.Max(x, 0));
            Assert.IsTrue(a == b, "ReLU Activation successful");
        }

        [TestMethod]
        public void ReLUPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new ReLU().Backward(a);
            b.InMap((x) => (x > 0) ? 1 : 0);
            Assert.IsTrue(a == b, "ReLU Derivative successful");
        }

        [TestMethod]
        public void SeLUTest()
        {
            double Alpha = 1.6732632423543772848170429916717;
            double Lambda = 1.0507009873554804934193349852946;

            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new SeLU().Forward(a);
            b.InMap((x) => (x > 0) ? x : Alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(a == b, "SeLU Activation successful");
        }

        [TestMethod]
        public void SeLUPrimeTest()
        {
            double Alpha = 1.6732632423543772848170429916717;
            double Lambda = 1.0507009873554804934193349852946;

            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new SeLU().Backward(a);
            b.InMap((x) => (x > 0) ? Lambda : Lambda * Alpha * Math.Exp(x));
            Assert.IsTrue(a == b, "SeLU Derivative successful");
        }

        [TestMethod]
        public void SigmoidTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Sigmoid().Forward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)));
            Assert.IsTrue(a == b, "Sigmoid Activation successful");
        }

        [TestMethod]
        public void SigmoidPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Sigmoid().Backward(a);
            b.InMap((x) => (1.0 / (1 + Math.Exp(-x))) * (1 - (1.0 / (1 + Math.Exp(-x)))));
            Assert.IsTrue(a == b, "Sigmoid Derivative successful");
        }

        [TestMethod]
        public void SoftmaxTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();

            double sumExp = 0.0;
            Matrix res = b.Duplicate();
            sumExp = 0.0;

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    sumExp += Math.Exp(b[i, j]);
                }
            }

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    res[i, j] = Math.Exp(b[i, j]) / sumExp;
                }
            }

            b = res;
            a = new Softmax().Forward(a);
            Assert.IsTrue(a == b, "Softmax Activation successful");
        }

        [TestMethod]
        public void SoftmaxPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();

            double sumExp = 0.0;
            Matrix res = b.Duplicate();
            sumExp = 0.0;

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    sumExp += Math.Exp(b[i, j]);
                }
            }

            for (int i = 0; i < res.Rows; i++)
            {
                for (int j = 0; j < res.Columns; j++)
                {
                    res[i, j] = Math.Exp(b[i, j]) / sumExp;
                }
            }

            b = res;

            Softmax s = new Softmax();
            a = s.Forward(a);
            a = s.Backward(a);

            b.InMap((x) => Math.Exp(x) / sumExp * (1 - Math.Exp(x) / sumExp));

            Assert.IsTrue(a == b, "Softmax Derivative successful");
        }

        [TestMethod]
        public void SoftplusTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Softplus().Forward(a);
            b.InMap((x) => Math.Log(1 + Math.Exp(x)));
            Assert.IsTrue(a == b, "Softplus Activation successful");
        }

        [TestMethod]
        public void SoftplusPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Softplus().Backward(a);
            b.InMap((x) => Math.Exp(x) / (1 + Math.Exp(x)));
            Assert.IsTrue(a == b, "Softplus Derivative successful");
        }

        [TestMethod]
        public void SoftsignTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Softsign().Forward(a);
            b.InMap((x) => x / (1 + System.Math.Abs(x)));
            Assert.IsTrue(a == b, "Softsign Activation successful");
        }

        [TestMethod]
        public void SoftsignPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Softsign().Backward(a);
            b.InMap((x) => x / System.Math.Pow((1 + System.Math.Abs(x)), 2));
            Assert.IsTrue(a == b, "Softsign Derivative successful");
        }

        [TestMethod]
        public void TanhTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Tanh().Forward(a);
            b.InMap((x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
            Assert.IsTrue(a == b, "Tanh Activation successful");
        }

        [TestMethod]
        public void TanhPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();
            a = new Tanh().Backward(a);
            b.InMap((x) => 1 - Math.Pow((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)), 2));
            Assert.IsTrue(a == b, "Tanh Derivative successful");
        }
    }
}
