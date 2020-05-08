// Copyright © 2020 Void-Intelligence All Rights Reserved.

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
using Vortex.Initializers.Kernels;

namespace VortexTests
{
    [TestClass]
    public class VortexNetwork
    {
        [TestMethod]
        public void SequentualForwardTest()
        {
            var network = new Network(new CrossEntropyCost(), new GradientDescent(0.1));
            network.CreateLayer(ELayerType.FullyConnected, 784, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.FullyConnected, 100, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.Output, 10, new Sigmoid(), new L2(2), new Normal());
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
            network.CreateLayer(ELayerType.FullyConnected, 784, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.FullyConnected, 100, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.Dropout, 100, new Sigmoid(), new L2(2), new Normal());
            network.CreateLayer(ELayerType.Output, 10, new Sigmoid(), new L2(2), new Normal());
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
            net.CreateLayer(ELayerType.FullyConnected, 3, new Tanh(), new None(), new Normal(), 0.01);
            net.CreateLayer(ELayerType.FullyConnected, 25, new Tanh(), new None(), new Normal(), 0.01);
            net.CreateLayer(ELayerType.FullyConnected, 25, new Tanh(), new None(), new Normal(), 0.01);
            net.CreateLayer(ELayerType.Output, 1, new Tanh(), new None(), new Normal(), 0.01);
            net.InitNetwork();

            var inputs = new List<Matrix>();
            var outputs = new List<Matrix>();

            var m = new Matrix(3, 1);
            var n = new Matrix(1, 1);

            // 0 0 0    => 0
            m[0, 0] = 0; m[1, 0] = 0; m[2, 0] = 0;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 0 0 1    => 1
            m[0, 0] = 0; m[1, 0] = 0; m[2, 0] = 1;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 0 1 0    => 1
            m[0, 0] = 0; m[1, 0] = 1; m[2, 0] = 0;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 0 1 1    => 0
            m[0, 0] = 0; m[1, 0] = 1; m[2, 0] = 1;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 0 0    => 1
            m[0, 0] = 1; m[1, 0] = 0; m[2, 0] = 0;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 0 1    => 0
            m[0, 0] = 1; m[1, 0] = 0; m[2, 0] = 1;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 1 0    => 0
            m[0, 0] = 1; m[1, 0] = 1; m[2, 0] = 0;
            n[0, 0] = 0;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            // 1 1 1    => 1
            m[0, 0] = 1; m[1, 0] = 1; m[2, 0] = 1;
            n[0, 0] = 1;
            inputs.Add(m.Duplicate());
            outputs.Add(n.Duplicate());

            for (var i = 0; i < 5000; i++)
            {
                net.Forward(inputs[0]);
                net.Backward(outputs[0]);

                net.Forward(inputs[1]);
                net.Backward(outputs[1]);

                net.Forward(inputs[2]);
                net.Backward(outputs[2]);

                net.Forward(inputs[3]);
                net.Backward(outputs[3]);

                net.Forward(inputs[4]);
                net.Backward(outputs[4]);

                net.Forward(inputs[5]);
                net.Backward(outputs[5]);

                net.Forward(inputs[6]);
                net.Backward(outputs[6]);

                net.Forward(inputs[7]);
                net.Backward(outputs[7]);

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
