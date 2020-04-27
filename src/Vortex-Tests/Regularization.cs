using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Regularization;

namespace VortexTests
{
    [TestClass]
    public class VortexRegularization
    {
        [TestMethod]
        public void L1Test()
        {
            double lambda = 2.0;
            L1 reg = new L1(new L1Settings(lambda));
            Matrix mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs(mat.AbsoluteNorm() * lambda - reg.CalculateNorm(mat)) < 0.1, "L1 Norm calculation successful");
        }

        [TestMethod]
        public void L2Test()
        {
            double lambda = 2.0;
            L2 reg = new L2(new L2Settings(lambda));
            Matrix mat = new Matrix(10, 10);
            mat.InRandomize();

            Assert.IsTrue(Math.Abs(mat.FrobeniusNorm() * lambda - reg.CalculateNorm(mat)) < 0.1, "L1 Norm calculation successful");

        }
    }
}
