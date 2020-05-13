// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Activation.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Activation
    {
        [TestMethod]
        public void ArctanTest()
        
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Arctan().Forward(a);
            b.InMap(Math.Atan);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Arctan Activation.");
        }

        [TestMethod]
        public void ArctanPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Arctan().Backward(a);
            b.InMap((x) => 1 / (1 + Math.Pow(x, 2)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Arctan Derivative.");
        }

        [TestMethod]
        public void BinaryStepTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BinaryStep().Forward(a);
            b.InMap((x) => x < 0 ? 0 : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "BinaryStep Activation.");
        }

        [TestMethod]
        public void BinaryStepPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BinaryStep().Backward(a);
            b.InMap((x) => 0);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "BinaryStep Derivative.");
        }

        [TestMethod]
        public void BipolarSigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BipolarSigmoid().Forward(a);
            b.InMap((x) => -1 + 2 / (1 + Math.Exp(-x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Bipolar Sigmoid Activation.");
        }

        [TestMethod]
        public void BipolarSigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BipolarSigmoid().Backward(a);
            b.InMap((x) => 0.5 * (1 + (-1 + 2 / (1 + Math.Exp(-x)))) * (1 - (-1 + 2 / (1 + Math.Exp(-x)))));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Bipolar Sigmoid Derivative.");
        }

        [TestMethod]
        public void EluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            const double alpha = 0.01;
            a = new Elu(0.01).Forward(a);
            b.InMap((x) => x >= 0 ? x : alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "ELU Activation.");
        }

        [TestMethod]
        public void EluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            const double alpha = 0.01;
            a = new Elu(0.01).Backward(a);
            b.InMap((x) => x >= 0 ? 1 : alpha * Math.Exp(x));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "ELU Derivative.");
        }

        [TestMethod]
        public void ExponentialTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Exponential().Forward(a);
            b.InMap(Math.Exp);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Exponential Activation.");
        }

        [TestMethod]
        public void ExponentialPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Exponential().Backward(a);
            b.InMap(Math.Exp);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Exponential Derivative.");
        }

        [TestMethod]
        public void HardSigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardSigmoid().Forward(a);
            b.InMap((x) => x < 0 ? 0 : x < 1 ? x : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Hard Sigmoid Activation.");
        }

        [TestMethod]
        public void HardSigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardSigmoid().Backward(a);
            b.InMap((x) => x > 1 || x < 0 ? 0 : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Hard Sigmoid Derivative.");
        }

        [TestMethod]
        public void HardTanhTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardTanh().Forward(a);
            b.InMap((x) => x < -1 ? -1 : x > 1 ? 1 : x);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Hard Tanh Activation.");
        }

        [TestMethod]
        public void HardTanhPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardTanh().Backward(a);
            b.InMap((x) => x < -1 ? 0 : x > 1 ? 0 : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Hard Tanh Derivative.");
        }

        [TestMethod]
        public void IdentityTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Identity().Forward(a);
            b.InMap((x) => x);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Identity Activation.");
        }

        [TestMethod]
        public void IdentityPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Identity().Backward(a);
            b.InMap((x) => 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Identity Derivative.");
        }

        [TestMethod]
        public void LoggyTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Loggy().Forward(a);
            b.InMap((x) => Math.Tanh(x / 2.0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Loggy Activation.");
        }

        [TestMethod]
        public void LoggyPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Loggy().Backward(a);
            b.InMap((x) => 1.0 / (Math.Cosh(x) + 1.0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Loggy Derivative.");
        }

        [TestMethod]
        public void LogitTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Logit().Forward(a);
            b.InMap((x) => Math.Log(x / (1 - x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Logit Activation.");
        }

        [TestMethod]
        public void LogitPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Logit().Backward(a);
            b.InMap((x) => -1 / Math.Pow(x, 2) - 1 / Math.Pow(1 - x, 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Logit Derivative.");
        }

        [TestMethod]
        public void LReluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            var Alpha = 0.01;
            a = new LRelu(0.01).Forward(a);
            b.InMap((x) => Math.Max(x, Alpha));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "ReLU Activation.");
        }

        [TestMethod]
        public void LReluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            var Alpha = 0.01;
            a = new LRelu(0.01).Backward(a);
            b.InMap((x) => x > 0 ? 1 : Alpha);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "ReLU Derivative.");
        }

        [TestMethod]
        public void MishTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Mish().Forward(a);
            b.InMap((x) => x * Math.Tanh(Math.Log(1 + Math.Exp(x))));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Mish Activation.");
        }

        [TestMethod]
        public void MishPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Mish().Backward(a);
            b.InMap((x) => Math.Exp(x) * (4 * (x + 1) + 4 * Math.Exp(2 * x) + Math.Exp(3 * x) + Math.Exp(x) * (4 * x + 6)) / Math.Pow(2 * Math.Exp(x) + Math.Exp(2 * x) + 2, 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Mish Derivative.");
        }

        [TestMethod]
        public void ReluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Relu().Forward(a);
            b.InMap((x) => Math.Max(x, 0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "ReLU Activation.");
        }

        [TestMethod]
        public void ReluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Relu().Backward(a);
            b.InMap((x) => x > 0 ? 1 : 0);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "ReLU Derivative.");
        }

        [TestMethod]
        public void SeluTest()
        {
            var Alpha = 1.6732632423543772848170429916717;

            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Selu().Forward(a);
            b.InMap((x) => x > 0 ? x : Alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "SeLU Activation.");
        }

        [TestMethod]
        public void SeluPrimeTest()
        {
            var Alpha = 1.6732632423543772848170429916717;
            var Lambda = 1.0507009873554804934193349852946;

            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Selu().Backward(a);
            b.InMap((x) => x > 0 ? Lambda : Lambda * Alpha * Math.Exp(x));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "SeLU Derivative.");
        }

        [TestMethod]
        public void SigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Sigmoid().Forward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Sigmoid Activation.");
        }

        [TestMethod]
        public void SigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Sigmoid().Backward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)) * (1 - 1.0 / (1 + Math.Exp(-x))));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Sigmoid Derivative.");
        }

        [TestMethod]
        public void SoftmaxTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();

            var res = b.Duplicate();
            var sumExp = 0.0;

            for (var i = 0; i < res.Rows; i++)
            for (var j = 0; j < res.Columns; j++) sumExp += Math.Exp(b[i, j]);

            for (var i = 0; i < res.Rows; i++)
            for (var j = 0; j < res.Columns; j++) res[i, j] = Math.Exp(b[i, j]) / sumExp;

            b = res;
            a = new Softmax().Forward(a);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Softmax Activation.");
        }

        [TestMethod]
        public void SoftmaxPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();

            var sumExp = 0.0;

            for (var i = 0; i < b.Rows; i++)
            for (var j = 0; j < b.Columns; j++) sumExp += Math.Exp(a[i, j]);

            for (var i = 0; i < b.Rows; i++)
            for (var j = 0; j < b.Columns; j++)
                b[i, j] = Math.Exp(a[i, j]) / sumExp * (1.0 - Math.Exp(a[i, j]) / sumExp);
            
            var s = new Softmax();
            a = s.Backward(a);

            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Softmax Derivative.");
        }

        [TestMethod]
        public void SoftplusTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softplus().Forward(a);
            b.InMap((x) => Math.Log(1 + Math.Exp(x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Softplus Activation.");
        }

        [TestMethod]
        public void SoftplusPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softplus().Backward(a);
            b.InMap((x) => Math.Exp(x) / (1 + Math.Exp(x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Softplus Derivative.");
        }

        [TestMethod]
        public void SoftsignTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softsign().Forward(a);
            b.InMap((x) => x / (1 + Math.Abs(x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Softsign Activation.");
        }

        [TestMethod]
        public void SoftsignPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softsign().Backward(a);
            b.InMap((x) => x / Math.Pow(1 + Math.Abs(x), 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Softsign Derivative.");
        }

        [TestMethod]
        public void SwishTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Swish().Forward(a);
            b.InMap((x) => Math.Exp(-x) + 1.0);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Tanh Activation.");
        }

        [TestMethod]
        public void SwishPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Swish().Backward(a);
            b.InMap((x) => (1.0 + Math.Exp(x) + x) / Math.Pow(1.0 + Math.Exp(x), 2.0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Tanh Derivative.");
        }

        [TestMethod]
        public void TanhTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Tanh().Forward(a);
            b.InMap((x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Tanh Activation.");
        }

        [TestMethod]
        public void TanhPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Tanh().Backward(a);
            b.InMap((x) => 1 - Math.Pow((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)), 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, "Tanh Derivative.");
        }
    }
}
