﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Nomad.Matrix;
using Vortex.Network;
using Vortex.Layer.Utility;
using Vortex.Activation.Kernels;
using Vortex.Regularization.Kernels;
using Vortex.Cost.Kernels;
using Vortex.Optimizer.Kernels;
using Vortex.Initializer.Kernels;
using Vortex.Mutation.Kernels;

namespace VortexTests
{
    [TestClass]
    public class VortexNetwork
    {
        [TestMethod]
        public void SequentualForwardTest()
        {
            var network = new Network(new CrossEntropyCost(), new GradientDescent(0.1));
            network.CreateLayer(ELayerType.FullyConnected, 784, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.FullyConnected, 100, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.Output, 10, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.InitNetwork();

            var x = new Matrix(784, 1);
            x.InFlatten();

            var y = new Matrix(10, 1);
            y.InRandomize();

            network.Forward(x);
        }

        [TestMethod]
        public void SequentualBackwardTest()
        {
            var network = new Network(new CrossEntropyCost(), new GradientDescent(0.1));
            network.CreateLayer(ELayerType.FullyConnected, 784, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.FullyConnected, 100, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.CreateLayer(ELayerType.Output, 10, new Sigmoid(), new L2(2), new Normal(), new NoMutation());
            network.InitNetwork();

            var x = new Matrix(784, 1);
            x.InFlatten();

            var y = new Matrix(10, 1);
            y.InRandomize();

            network.Forward(x);
            network.Backward(y);
        }


        [TestMethod]
        public void XorTest()
        {
            var net = new Network(new QuadraticCost(), new GradientDescent(0.03)); 
            net.CreateLayer(ELayerType.FullyConnected, 3, new Tanh(), new None(), new Normal(), new NoMutation(), 0.01);
            net.CreateLayer(ELayerType.FullyConnected, 25, new Tanh(), new None(), new Normal(), new NoMutation(), 0.01);
            net.CreateLayer(ELayerType.FullyConnected, 25, new Tanh(), new None(), new Normal(), new NoMutation(), 0.01);
            net.CreateLayer(ELayerType.Output, 1, new Tanh(), new None(), new Normal(), new NoMutation(), 0.01);
            net.InitNetwork();

            var inputs = new List<Matrix>();
            var outputs = new List<Matrix>();

            // 0 0 0    => 0
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 0.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 0.0 } }));

            // 0 0 1    => 1
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 0.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 0 1 0    => 1
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 1.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 0 1 1    => 0
            inputs.Add(new Matrix(new double[,] { { 0.0 }, { 1.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 1 0 0    => 1
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 0.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            // 1 0 1    => 0
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 0.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 0.0 } }));

            // 1 1 0    => 0
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 1.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new double[,] { { 0.0 } }));

            // 1 1 1    => 1
            inputs.Add(new Matrix(new double[,] { { 1.0 }, { 1.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new double[,] { { 1.0 } }));

            for (var i = 0; i < 5000; i++)
            {
                net.Train(inputs[i % 8], outputs[i % 8]);
            }

            var correct = 0;
            for (var i = 0; i < 125; i++)
            {
                correct += Math.Abs(net.Forward(inputs[0])[0, 0]) < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[1])[0, 0]) - 1 < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[2])[0, 0]) - 1 < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[3])[0, 0]) < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[4])[0, 0]) - 1 < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[5])[0, 0]) < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[6])[0, 0]) < 0.1 ? 1 : 0;
                correct += Math.Abs(net.Forward(inputs[7])[0, 0]) - 1 < 0.1 ? 1 : 0;
            }
            var acc = correct / 1000.0 * 100.0;

            Trace.WriteLine(" Acc: " + acc);
            Assert.IsTrue(acc > 80.0, "Network did not learn XOR");
        }
    }
}
