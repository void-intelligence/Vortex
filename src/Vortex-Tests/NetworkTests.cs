// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Nomad.Matrix;
using Vortex.Network;
using Vortex.Activation.Kernels;
using Vortex.Cost.Kernels.Legacy;
using Vortex.Initializer.Kernels;
using Vortex.Optimizer.Kernels;
using Vortex.Layer.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Network
    {
        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException), "Network is Locked.")]
        public void NetworkLockedExceptionInit()
        {
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));
            net.InitNetwork();
            net.InitNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException), "Network is Locked.")]
        public void NetworkLockedExceptionLayer()
        {
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));
            net.InitNetwork();
            net.CreateLayer(new FullyConnected(3, new Tanh()));
        }

        [TestMethod]
        public void SequentialXor()
        {
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh(), null,new HeUniform()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));
            net.InitNetwork();

            _ = net.Y;

            var inputs = new List<Matrix>();
            var outputs = new List<Matrix>();

            // 0 0 0    => 0
            inputs.Add(new Matrix(new[,] { { 0.0 }, { 0.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new[,] { { 0.0 } }));

            // 0 0 1    => 1
            inputs.Add(new Matrix(new[,] { { 0.0 }, { 0.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new[,] { { 1.0 } }));

            // 0 1 0    => 1
            inputs.Add(new Matrix(new[,] { { 0.0 }, { 1.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new[,] { { 1.0 } }));

            // 0 1 1    => 0
            inputs.Add(new Matrix(new[,] { { 0.0 }, { 1.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new[,] { { 1.0 } }));

            // 1 0 0    => 1
            inputs.Add(new Matrix(new[,] { { 1.0 }, { 0.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new[,] { { 1.0 } }));

            // 1 0 1    => 0
            inputs.Add(new Matrix(new[,] { { 1.0 }, { 0.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new[,] { { 0.0 } }));

            // 1 1 0    => 0
            inputs.Add(new Matrix(new[,] { { 1.0 }, { 1.0 }, { 0.0 } }));
            outputs.Add(new Matrix(new[,] { { 0.0 } }));

            // 1 1 1    => 1
            inputs.Add(new Matrix(new[,] { { 1.0 }, { 1.0 }, { 1.0 } }));
            outputs.Add(new Matrix(new[,] { { 1.0 } }));

            for (var i = 0; i < 8; i++) net.Train(inputs[i % 8], outputs[i % 8]);

            var correct = 0;
            for (var i = 0; i < 10; i++)
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
            var acc = correct / 80.0 * 100.0;

            Trace.WriteLine(" Acc: " + acc);
            Assert.IsTrue(acc > 80.0, "Network did not learn XOR");
        }
    }
}
