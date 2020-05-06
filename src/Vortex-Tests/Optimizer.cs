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

            for (int i = 0; i < 1000; i++)
            {
                dJdX.InRandomize();
                Matrix deltas = gd.CalculateDeltaW(X, dJdX);
                X -= deltas;
            }
        }

        [TestMethod]
        public void MomentumTest()
        {
            Momentum momentum = new Momentum();
            Matrix X = new Matrix(10, 10);
            Matrix dJdX = new Matrix(10, 10);

            X.InFill(100);

            for (int i = 0; i < 1000; i++)
            {
                dJdX.InRandomize();
                Matrix deltas = momentum.CalculateDeltaW(X, dJdX);
                X -= deltas;
            }
        }

        [TestMethod]
        public void RMSPropTest()
        {
            RMSProp rms = new RMSProp();
            Matrix X = new Matrix(10, 10);
            Matrix dJdX = new Matrix(10, 10);

            X.InFill(100);
           

            for (int i = 0; i < 1000; i++)
            {
                dJdX.InRandomize(2);
                Matrix deltas = rms.CalculateDeltaW(X, dJdX);
                X -= deltas;
            }
        }
    }
}
