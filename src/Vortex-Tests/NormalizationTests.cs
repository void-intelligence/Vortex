// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Normalization.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Normalization
    {
        [TestMethod]
        public void NoNormalizationTest()
        {
            var a = new Matrix(5, 5);
            a.InRandomize();
            var norm = new NoNorm();
            Assert.IsTrue(Math.Abs(a.FrobeniusNorm() - norm.Normalize(a).FrobeniusNorm()) < 0.1, norm.Type().ToString() + " Normalization!");
        }
    }
}
