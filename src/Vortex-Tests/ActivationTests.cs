// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Activation.Kernels;

namespace VortexTests
{
    [TestClass]
    public class VortexActivation
    {
        [TestMethod]
        public void ArctanTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new ArctanKernel().Forward(a);
            b.InMap(Math.Atan);
            Assert.IsTrue(a == b, "Arctan Activation successful");
        }

        [TestMethod]
        public void ArctanPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new ArctanKernel().Backward(a);
            b.InMap((x) => 1 / (1 + Math.Pow(x, 2)));
            Assert.IsTrue(a == b, "Arctan Derivative successful");
        }

        [TestMethod]
        public void BinaryStepTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BinaryStepKernel().Forward(a);
            b.InMap((x) => x < 0 ? 0 : 1);
            Assert.IsTrue(a == b, "BinaryStep Activation successful");
        }

        [TestMethod]
        public void BinaryStepPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BinaryStepKernel().Backward(a);
            b.InMap((x) => 0);
            Assert.IsTrue(a == b, "BinaryStep Derivative successful");
        }

        [TestMethod]
        public void BipolarSigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BipolarSigmoidKernel().Forward(a);
            b.InMap((x) => -1 + 2 / (1 + Math.Exp(-x)));
            Assert.IsTrue(a == b, "Bipolar Sigmoid Activation successful");
        }

        [TestMethod]
        public void BipolarSigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BipolarSigmoidKernel().Backward(a);
            b.InMap((x) => 0.5 * (1 + (-1 + 2 / (1 + Math.Exp(-x)))) * (1 - (-1 + 2 / (1 + Math.Exp(-x)))));
            Assert.IsTrue(a == b, "Bipolar Sigmoid Derivative successful");
        }

        [TestMethod]
        public void EluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            var Alpha = 0.01;
            a = new EluKernel(new Elu()).Forward(a);
            b.InMap((x) => x >= 0 ? x : Alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(a == b, "ELU Activation successful");
        }

        [TestMethod]
        public void EluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            var Alpha = 0.01;
            a = new EluKernel(new Elu()).Backward(a);
            b.InMap((x) => x >= 0 ? 1 : Alpha * Math.Exp(x));
            Assert.IsTrue(a == b, "ELU Derivative successful");
        }

        [TestMethod]
        public void HardSigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardSigmoidKernel().Forward(a);
            b.InMap((x) => x < 0 ? 0 : x < 1 ? x : 1);
            Assert.IsTrue(a == b, "Hard Sigmoid Activation successful");
        }

        [TestMethod]
        public void HardSigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardSigmoidKernel().Backward(a);
            b.InMap((x) => x > 1 || x < 0 ? 0 : 1);
            Assert.IsTrue(a == b, "Hard Sigmoid Derivative successful");
        }

        [TestMethod]
        public void HardTanhTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardTanhKernel().Forward(a);
            b.InMap((x) => x < -1 ? -1 : x > 1 ? 1 : x);
            Assert.IsTrue(a == b, "Hard Tanh Activation successful");
        }

        [TestMethod]
        public void HardTanhPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardTanhKernel().Backward(a);
            b.InMap((x) => x < -1 ? 0 : x > 1 ? 0 : 1);
            Assert.IsTrue(a == b, "Hard Tanh Derivative successful");
        }

        [TestMethod]
        public void IdentityTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new IdentityKernel().Forward(a);
            b.InMap((x) => x);
            Assert.IsTrue(a == b, "Identity Activation successful");
        }

        [TestMethod]
        public void IdentityPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new IdentityKernel().Backward(a);
            b.InMap((x) => 1);
            Assert.IsTrue(a == b, "Identity Derivative successful");
        }

        [TestMethod]
        public void LogitTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new LogitKernel().Forward(a);
            b.InMap((x) => Math.Log(x / (1 - x)));
            Assert.IsTrue(a == b, "Logit Activation successful");
        }

        [TestMethod]
        public void LogitPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new LogitKernel().Backward(a);
            b.InMap((x) => -1 / Math.Pow(x, 2) - 1 / Math.Pow(1 - x, 2));
            Assert.IsTrue(a == b, "Logit Derivative successful");
        }

        [TestMethod]
        public void LReluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            var Alpha = 0.01;
            a = new LReluKernel(new LRelu()).Forward(a);
            b.InMap((x) => Math.Max(x, Alpha));
            Assert.IsTrue(a == b, "ReLU Activation successful");
        }

        [TestMethod]
        public void LReluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            var Alpha = 0.01;
            a = new LReluKernel(new LRelu()).Backward(a);
            b.InMap((x) => x > 0 ? 1 : Alpha);
            Assert.IsTrue(a == b, "ReLU Derivative successful");
        }

        [TestMethod]
        public void MishTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new MishKernel().Forward(a);
            b.InMap((x) => x * Math.Tanh(Math.Log(1 + Math.Exp(x))));
            Assert.IsTrue(a == b, "Mish Activation successful");
        }

        [TestMethod]
        public void MishPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new MishKernel().Backward(a);
            b.InMap((x) => Math.Exp(x) * (4 * (x + 1) + 4 * Math.Exp(2 * x) + Math.Exp(3 * x) + Math.Exp(x) * (4 * x + 6)) / Math.Pow(2 * Math.Exp(x) + Math.Exp(2 * x) + 2, 2));
            Assert.IsTrue(a == b, "Mish Derivative successful");
        }

        [TestMethod]
        public void ReluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new ReluKernel().Forward(a);
            b.InMap((x) => Math.Max(x, 0));
            Assert.IsTrue(a == b, "ReLU Activation successful");
        }

        [TestMethod]
        public void ReluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new ReluKernel().Backward(a);
            b.InMap((x) => x > 0 ? 1 : 0);
            Assert.IsTrue(a == b, "ReLU Derivative successful");
        }

        [TestMethod]
        public void SeluTest()
        {
            var Alpha = 1.6732632423543772848170429916717;

            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SeluKernel().Forward(a);
            b.InMap((x) => x > 0 ? x : Alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(a == b, "SeLU Activation successful");
        }

        [TestMethod]
        public void SeluPrimeTest()
        {
            var Alpha = 1.6732632423543772848170429916717;
            var Lambda = 1.0507009873554804934193349852946;

            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SeluKernel().Backward(a);
            b.InMap((x) => x > 0 ? Lambda : Lambda * Alpha * Math.Exp(x));
            Assert.IsTrue(a == b, "SeLU Derivative successful");
        }

        [TestMethod]
        public void SigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SigmoidKernel().Forward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)));
            Assert.IsTrue(a == b, "Sigmoid Activation successful");
        }

        [TestMethod]
        public void SigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SigmoidKernel().Backward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)) * (1 - 1.0 / (1 + Math.Exp(-x))));
            Assert.IsTrue(a == b, "Sigmoid Derivative successful");
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
            a = new SoftmaxKernel().Forward(a);
            Assert.IsTrue(a == b, "Softmax Activation successful");
        }

        [TestMethod]
        public void SoftmaxPrimeTest()
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

            var s = new SoftmaxKernel();
            a = s.Forward(a);
            a = s.Backward(a);

            b.InMap((x) => Math.Exp(x) / sumExp * (1 - Math.Exp(x) / sumExp));

            Assert.IsTrue(a == b, "Softmax Derivative successful");
        }

        [TestMethod]
        public void SoftplusTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SoftplusKernel().Forward(a);
            b.InMap((x) => Math.Log(1 + Math.Exp(x)));
            Assert.IsTrue(a == b, "Softplus Activation successful");
        }

        [TestMethod]
        public void SoftplusPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SoftplusKernel().Backward(a);
            b.InMap((x) => Math.Exp(x) / (1 + Math.Exp(x)));
            Assert.IsTrue(a == b, "Softplus Derivative successful");
        }

        [TestMethod]
        public void SoftsignTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SoftsignKernel().Forward(a);
            b.InMap((x) => x / (1 + Math.Abs(x)));
            Assert.IsTrue(a == b, "Softsign Activation successful");
        }

        [TestMethod]
        public void SoftsignPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new SoftsignKernel().Backward(a);
            b.InMap((x) => x / Math.Pow(1 + Math.Abs(x), 2));
            Assert.IsTrue(a == b, "Softsign Derivative successful");
        }

        [TestMethod]
        public void TanhTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new TanhKernel().Forward(a);
            b.InMap((x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
            Assert.IsTrue(a == b, "Tanh Activation successful");
        }

        [TestMethod]
        public void TanhPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new TanhKernel().Backward(a);
            b.InMap((x) => 1 - Math.Pow((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)), 2));
            Assert.IsTrue(a == b, "Tanh Derivative successful");
        }
    }
}
