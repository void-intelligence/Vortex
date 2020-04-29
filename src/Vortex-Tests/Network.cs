// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using System.Collections.Generic;
using Vortex.Activation;
using Vortex.Cost;
using Vortex.Layer.Utility;
using Vortex.Network;
using Vortex.Optimizer;
using Vortex.Regularization;

namespace VortexTests
{
    [TestClass]
    public class VortexNetwork
    {
        [TestMethod]
        public void SequentualForwardTest()
        {
            Network network = new Network(new CrossEntropyCostSettings(), new GradientDescentSettings(0.001));
            network.CreateLayer(ELayerType.FullyConnected, 784, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.Dropout, 100, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.FullyConnected, 100, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.Dropout, 100, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.Output, 10, new SigmoidSettings(), new L2Settings(2));
            network.InitNetwork();

            Matrix x = new Matrix(784, 1);
            x.InFlatten();

            Matrix y = new Matrix(10, 1);
            y.InRandomize();

            network.Forward(x, y);
        }

        [TestMethod]
        public void SequentualBackwardTest()
        {
            Network network = new Network(new CrossEntropyCostSettings(), new GradientDescentSettings(0.001));
            network.CreateLayer(ELayerType.FullyConnected, 784, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.Dropout, 100, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.FullyConnected, 100, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.Dropout, 100, new SigmoidSettings(), new L2Settings(2));
            network.CreateLayer(ELayerType.Output, 10, new SigmoidSettings(), new L2Settings(2));
            network.InitNetwork();

            Matrix x = new Matrix(784, 1);
            x.InFlatten();

            Matrix y = new Matrix(10, 1);
            y.InRandomize();

            network.Forward(x, y);
            network.Backward();
        }

        [TestMethod]
        public void XORTest()
        {
            Network network = new Network(new QuadraticCostSettings(), new GradientDescentSettings(0.01));
            network.CreateLayer(ELayerType.FullyConnected, 2, new ReLUSettings(), new NoneSettings());
            network.CreateLayer(ELayerType.FullyConnected, 20, new ReLUSettings(), new NoneSettings());
            network.CreateLayer(ELayerType.Output, 1, new ReLUSettings(), new NoneSettings());
            network.InitNetwork();

            List<Matrix> inputs = new List<Matrix>();
            List<Matrix> outputs = new List<Matrix>();

            Matrix m = new Matrix(2, 1);
            Matrix n = new Matrix(1, 1);

            m[0, 0] = 0; m[1, 0] = 0;
            inputs.Add(m.Duplicate());
            n[0, 0] = 0;
            outputs.Add(n.Duplicate());

            m[0, 0] = 0; m[1, 0] = 1;
            inputs.Add(m.Duplicate());
            n[0, 0] = 1;
            outputs.Add(n.Duplicate());

            m[0, 0] = 1; m[1, 0] = 0;
            inputs.Add(m.Duplicate());
            n[0, 0] = 1;
            outputs.Add(n.Duplicate());

            m[0, 0] = 1; m[1, 0] = 1;
            inputs.Add(m.Duplicate());
            n[0, 0] = 0;
            outputs.Add(n.Duplicate());

            Random rng = new Random();
            int k = rng.Next(0, 4); ;
            for (int i = 0; i < 1000000; i++)
            {
                network.Forward(inputs[k], outputs[k]);
                network.Backward();
                k = rng.Next(0, 4);
            }

            Matrix x = network.Forward(inputs[0], outputs[0]); // 0
            Matrix y = network.Forward(inputs[1], outputs[1]); // 1
            Matrix z = network.Forward(inputs[2], outputs[2]); // 1
            Matrix w = network.Forward(inputs[3], outputs[3]); // 0

            Assert.IsTrue(Math.Abs(x[0, 0]) < 0.1, "X successful");
            Assert.IsTrue(Math.Abs(y[0, 0] - 1) < 0.1, "y successful");
            Assert.IsTrue(Math.Abs(z[0, 0] - 1) < 0.1, "z successful");
            Assert.IsTrue(Math.Abs(w[0, 0]) < 0.1, "w successful");
        }
    }
}
