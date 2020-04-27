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
            GradientDescent gd = new GradientDescent();
            Matrix X = new Matrix(10, 10);
            Matrix dJdX = new Matrix(10, 10);

            X.InFill(100);

            for (int i = 0; i < 100; i++)
            {
                dJdX.InRandomize();
                Matrix deltas = gd.CalculateDeltaW(X, dJdX);
                X -= deltas;
            }

            Assert.IsTrue(X.FrobeniusNorm() < X.Fill(100).FrobeniusNorm(), "SGD Successful");
        }

        [TestMethod]
        public void MomentumTest()
        {
            Momentum momentum = new Momentum();
            Matrix X = new Matrix(10, 10);
            Matrix dJdX = new Matrix(10, 10);

            X.InFill(100);

            for (int i = 0; i < 100; i++)
            {
                dJdX.InRandomize();
                Matrix deltas = momentum.CalculateDeltaW(X, dJdX);
                X -= deltas;
            }

            Assert.IsTrue(X.FrobeniusNorm() < X.Fill(100).FrobeniusNorm(), "Momentum Successful");
        }

        [TestMethod]
        public void RMSPropTest()
        {
            Momentum rms = new Momentum();
            Matrix X = new Matrix(10, 10);
            Matrix dJdX = new Matrix(10, 10);

            X.InFill(100);
           

            for (int i = 0; i < 100; i++)
            {
                dJdX.InRandomize();
                Matrix deltas = rms.CalculateDeltaW(X, dJdX);
                X -= deltas;
            }

            Assert.IsTrue(X.FrobeniusNorm() < X.Fill(100).FrobeniusNorm(), "RMSProp Successful");
        }
    }
}
