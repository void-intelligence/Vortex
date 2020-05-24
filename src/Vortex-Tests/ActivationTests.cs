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
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Arctan().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void ArctanPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Arctan().Backward(a);
            b.InMap((x) => 1 / (1 + Math.Pow(x, 2)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Arctan().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void BinaryStepTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BinaryStep().Forward(a);
            b.InMap((x) => x < 0 ? 0 : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new BinaryStep().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void BinaryStepPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BinaryStep().Backward(a);
            b.InMap((x) => 0);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new BinaryStep().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void BipolarSigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BipolarSigmoid().Forward(a);
            b.InMap((x) => -1 + 2 / (1 + Math.Exp(-x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new BipolarSigmoid().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void BipolarSigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new BipolarSigmoid().Backward(a);
            b.InMap((x) => 0.5 * (1 + (-1 + 2 / (1 + Math.Exp(-x)))) * (1 - (-1 + 2 / (1 + Math.Exp(-x)))));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new BipolarSigmoid().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void EluTest()
        {
            var e = new Elu(0) {Alpha = 2};
            var p = e.Alpha;

            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            const double alpha = 0.01;
            a = new Elu(0.01).Forward(a);
            b.InMap((x) => x >= 0 ? x : alpha * (Math.Exp(x) - 1));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Elu(0.01).Type().ToString() + "ELU Activation.");
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
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Elu(0.01).Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void ExponentialTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Exponential().Forward(a);
            b.InMap(Math.Exp);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Exponential().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void ExponentialPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Exponential().Backward(a);
            b.InMap(Math.Exp);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Exponential().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void HardSigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardSigmoid().Forward(a);
            b.InMap((x) => x < 0 ? 0 : x < 1 ? x : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new HardSigmoid().Type().ToString() + "  Activation.");
        }

        [TestMethod]
        public void HardSigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardSigmoid().Backward(a);
            b.InMap((x) => x > 1 || x < 0 ? 0 : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new HardSigmoid().Type().ToString() + "  Derivative.");
        }

        [TestMethod]
        public void HardTanhTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardTanh().Forward(a);
            b.InMap((x) => x < -1 ? -1 : x > 1 ? 1 : x);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new HardTanh().Type().ToString() + "  Activation.");
        }

        [TestMethod]
        public void HardTanhPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new HardTanh().Backward(a);
            b.InMap((x) => x < -1 ? 0 : x > 1 ? 0 : 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new HardTanh().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void IdentityTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Identity().Forward(a);
            b.InMap((x) => x);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Identity().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void IdentityPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Identity().Backward(a);
            b.InMap((x) => 1);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Identity().Type().ToString() + "  Derivative.");
        }

        [TestMethod]
        public void LoggyTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Loggy().Forward(a);
            b.InMap((x) => Math.Tanh(x / 2.0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Loggy().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void LoggyPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Loggy().Backward(a);
            b.InMap((x) => 1.0 / (Math.Cosh(x) + 1.0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Loggy().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void LogitTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Logit().Forward(a);
            b.InMap((x) => Math.Log(x / (1 - x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Logit().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void LogitPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Logit().Backward(a);
            b.InMap((x) => -1 / Math.Pow(x, 2) - 1 / Math.Pow(1 - x, 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Logit().Type().ToString() + " Derivative.");
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
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new LRelu(0.01).Type().ToString() + " Activation.");
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
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new LRelu(0.01).Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void MishTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Mish().Forward(a);
            b.InMap((x) => x * Math.Tanh(Math.Log(1 + Math.Exp(x))));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Mish().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void MishPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Mish().Backward(a);
            b.InMap((x) =>
                Math.Exp(x) * (4 * (x + 1) + 4 * Math.Exp(2 * x) + Math.Exp(3 * x) + Math.Exp(x) * (4 * x + 6)) /
                Math.Pow(2 * Math.Exp(x) + Math.Exp(2 * x) + 2, 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Mish().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void ReluTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Relu().Forward(a);
            b.InMap((x) => Math.Max(x, 0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Relu().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void ReluPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Relu().Backward(a);
            b.InMap((x) => x > 0 ? 1 : 0);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Relu().Type().ToString() + " Derivative.");
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
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Selu().Type().ToString() + " Activation.");
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
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Selu().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void SigmoidTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Sigmoid().Forward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Sigmoid().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void SigmoidPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Sigmoid().Backward(a);
            b.InMap((x) => 1.0 / (1 + Math.Exp(-x)) * (1 - 1.0 / (1 + Math.Exp(-x))));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1,
                new Sigmoid().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void SoftmaxTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();

            var sumExp = a.Map(Math.Exp).Sum();
            var res = a.Map(Math.Exp) / sumExp;

            var y = new Softmax().Forward(a);

            Assert.IsTrue(res.ToString() == y.ToString(), new Softmax().Type().ToString() + " Activation.");
        }


        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException), "Cannot operate softmax on a single value!")]
        public void SoftMaxException()
        {
            new Softmax().Activate(2);
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException), "Cannot operate softmax derivative on a single value!")]
        public void SoftMaxDerivativeException()
        {
            new Softmax().Derivative(2);
        }

        [TestMethod]
        public void SoftmaxPrimeTest()
        {
            var a = new Matrix(4, 4);
            a.InRandomize();

            var sf = new Softmax();

            // Jacobian Matrix
            var cachedSoftmax = sf.Forward(a);
            a.InFlatten();
            var jac = new Matrix(a.Rows, a.Rows).Fill(0);
            for (var i = 0; i < cachedSoftmax.Rows; i++) jac[i, i] = a[i, 0];
            for (var i = 0; i < jac.Rows; i++)
            for (var j = 0; j < jac.Columns; j++)
                if (i == j) jac[i, j] = cachedSoftmax[i, 0] * (1 - cachedSoftmax[j, 0]);

            var b = jac  -a * a.T();
            Assert.IsTrue(Math.Abs(sf.Backward(a).FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Softmax().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void SoftplusTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softplus().Forward(a);
            b.InMap((x) => Math.Log(1 + Math.Exp(x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Softplus().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void SoftplusPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softplus().Backward(a);
            b.InMap((x) => Math.Exp(x) / (1 + Math.Exp(x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Softplus().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void SoftsignTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softsign().Forward(a);
            b.InMap((x) => x / (1 + Math.Abs(x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Softsign().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void SoftsignPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Softsign().Backward(a);
            b.InMap((x) => x / Math.Pow(1 + Math.Abs(x), 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Softsign().Type().ToString() + " Derivative.");
        }

        [TestMethod]
        public void SwishTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Swish().Forward(a);
            b.InMap((x) => Math.Exp(-x) + 1.0);
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Swish().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void SwishPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Swish().Backward(a);
            b.InMap((x) => (1.0 + Math.Exp(x) + x) / Math.Pow(1.0 + Math.Exp(x), 2.0));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Swish().Type().ToString() + "  Derivative.");
        }

        [TestMethod]
        public void TanhTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Tanh().Forward(a);
            b.InMap((x) => (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Tanh().Type().ToString() + " Activation.");
        }

        [TestMethod]
        public void TanhPrimeTest()
        {
            var a = new Matrix(2, 2);
            a.InRandomize();
            var b = a.Duplicate();
            a = new Tanh().Backward(a);
            b.InMap((x) => 1 - Math.Pow((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)), 2));
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - b.FrobeniusNorm()) < 0.1, new Tanh().Type().ToString() + " Derivative.");
        }
    }
}
