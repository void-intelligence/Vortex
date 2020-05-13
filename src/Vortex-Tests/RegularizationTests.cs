// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Regularization.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Regularization
    {
        [TestMethod]
        public void NoneTest()
        {
            var reg = new None();
            var mat = new Matrix(10, 10);
            mat.InFill(0);

            Assert.IsTrue(Math.Abs(mat.AbsoluteNorm() - reg.CalculateNorm(mat)) < 0.1, reg.Type().ToString() + " Regularization.");
        }


        [TestMethod]
        public void L1Test()
        {
            const double lambda = 1.0;
            var reg = new L1();
            var mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs(mat.AbsoluteNorm() * lambda - reg.CalculateNorm(mat)) < 0.1, reg.Type().ToString() + " Regularization.");
        }

        [TestMethod]
        public void L2Test()
        {
            const double lambda = 1.0;
            var reg = new L2();
            var mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs(mat.FrobeniusNorm() * lambda - reg.CalculateNorm(mat)) < 0.1, reg.Type().ToString() + " Regularization.");
        }

        [TestMethod]
        public void L1L2Test()
        {
            const double lambda = 1.0;
            var reg = new L1L2();
            var mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs((mat.AbsoluteNorm() + mat.FrobeniusNorm()) * lambda - reg.CalculateNorm(mat)) < 0.1, reg.Type().ToString() + " Regularization.");
        }
    }
}
