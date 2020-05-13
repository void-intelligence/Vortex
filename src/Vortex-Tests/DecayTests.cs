// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Vortex.Decay.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Decay
    {
        [TestMethod]
        public void ExponentialTest()
        {
            var e = new Exponential(1);
            e.IncrementEpoch();
            var x = e.CalculateAlpha(0.3);
            Assert.IsTrue(Math.Abs(x - 0.110) < 0.001, e.Type().ToString() + " Decay.");
        }

        [TestMethod]
        public void IterationBasedTest()
        {
            var e = new IterationBased(1);
            e.IncrementEpoch();
            var x = e.CalculateAlpha(0.3);
            Assert.IsTrue(Math.Abs(x - 0.15) < 0.001, e.Type().ToString() + " Decay.");
        }

        [TestMethod]
        public void MultiplicationTest()
        {
            var e = new Multiplication(1, 2, 2);
            e.IncrementEpoch();
            var x = e.CalculateAlpha(0.3);
            Assert.IsTrue(Math.Abs(x - 0.4242) < 0.001, e.Type().ToString() + " Decay.");
        }

        [TestMethod]
        public void NoneTest()
        {
            var e = new None();
            e.IncrementEpoch();
            var x = e.CalculateAlpha(0.3);
            Assert.IsTrue(Math.Abs(x - 0.3) < 0.001, e.Type().ToString() + " Decay.");
        }

        [TestMethod]
        public void SubtractionTest()
        {
            var e = new Subtraction(0.1,2);
            e.IncrementEpoch();
            var x = e.CalculateAlpha(0.3);
            Assert.IsTrue(Math.Abs(x - 0.25) < 0.001, e.Type().ToString() + " Decay.");
        }
    }
}
