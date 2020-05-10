﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Regularization.Kernels;

namespace VortexTests
{
    [TestClass]
    public class VortexRegularization
    {
        [TestMethod]
        public void L1Test()
        {
            var lambda = 2.0;
            var reg = new L1Kernel(new L1(lambda));
            var mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs(mat.AbsoluteNorm() * lambda - reg.CalculateNorm(mat)) < 0.1, "L1 Norm calculation successful");
        }

        [TestMethod]
        public void L2Test()
        {
            var lambda = 2.0;
            var reg = new L2Kernel(new L2(lambda));
            var mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs(mat.FrobeniusNorm() * lambda - reg.CalculateNorm(mat)) < 0.1, "L1 Norm calculation successful");

        }
    }
}
