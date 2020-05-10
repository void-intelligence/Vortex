// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Optimizer.Kernels;

namespace VortexTests
{
    [TestClass]
    public class VortexOptimizer
    {
        [TestMethod]
        public void SgdTest()
        {
            var gd = new GradientDescent();
            var x = new Matrix(10, 10);
            var dJdX = new Matrix(10, 10);

            x.InFill(100);

            for (var i = 0; i < 1000; i++)
            {
                dJdX.InRandomize();
                var deltas = gd.CalculateDelta(x, dJdX);
                x -= deltas;
            }
        }

        [TestMethod]
        public void MomentumTest()
        {
            var momentum = new Momentum();
            var x = new Matrix(10, 10);
            var dJdX = new Matrix(10, 10);

            x.InFill(100);

            for (var i = 0; i < 1000; i++)
            {
                dJdX.InRandomize();
                var deltas = momentum.CalculateDelta(x, dJdX);
                x -= deltas;
            }
        }

        [TestMethod]
        public void RmsPropTest()
        {
            var rms = new RmsProp();
            var x = new Matrix(10, 10);
            var dJdX = new Matrix(10, 10);

            x.InFill(100);

            for (var i = 0; i < 1000; i++)
            {
                dJdX.InRandomize(2);
                var deltas = rms.CalculateDelta(x, dJdX);
                x -= deltas;
            }
        }
    }
}
