// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Optimizer;

namespace VortexTests
{
    [TestClass]
    public class VortexOptimizer
    {
        [TestMethod]
        public void SGDTest()
        {
            GradientDescent gd = new GradientDescent(new GradientDescentSettings(0.001));
            Matrix X = new Matrix(10, 10);
            Matrix dJdX = new Matrix(10, 10);
            X.InRandomize();
            dJdX.InRandomize();
            gd.CalculateDelta(X, dJdX);
        } 
    }
}
