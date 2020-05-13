// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Activation.Kernels;
using Vortex.Cost.Kernels.Legacy;
using Vortex.Layer.Kernels;
using Vortex.Network;
using Vortex.Optimizer.Kernels;
using Vortex.Optimizer.Utility;

namespace VortexTests
{
    [TestClass]
    public class Optimizer
    {
        [TestMethod]
        public void DefaultTest()
        {
            var net = new Sequential(new QuadraticCost(), new DefaultOptimizer(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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

            for (var i = 0; i < 128; i++) net.Train(inputs[i % 8], outputs[i % 8]);

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

        [TestMethod]
        public void SgdTest()
        {
            var net = new Sequential(new QuadraticCost(), new GradientDescent(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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

            for (var i = 0; i < 128; i++) net.Train(inputs[i % 8], outputs[i % 8]);

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


        [TestMethod]
        public void AdamTest()
        {
            var net = new Sequential(new QuadraticCost(), new Adam(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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


        [TestMethod]
        public void MomentumTest()
        {
            var net = new Sequential(new QuadraticCost(), new Momentum(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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


        [TestMethod]
        public void NesterovMomentum()
        {
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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


        [TestMethod]
        public void RmsPropTest()
        {
            var net = new Sequential(new QuadraticCost(), new RmsProp(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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


        [TestMethod]
        public void AdaMaxTest()
        {
            var net = new Sequential(new QuadraticCost(), new AdaMax(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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

        [TestMethod]
        public void NadamTest()
        {
            var net = new Sequential(new QuadraticCost(), new Nadam(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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

        [TestMethod]
        public void AdaGradTest()
        {
            var net = new Sequential(new QuadraticCost(), new AdaGrad(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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

        [TestMethod]
        public void AdaDeltaTest()
        {
            var net = new Sequential(new QuadraticCost(), new AdaDelta(0.03));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new FullyConnected(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));

            net.InitNetwork();

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
