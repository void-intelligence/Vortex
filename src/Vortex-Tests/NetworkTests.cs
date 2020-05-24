// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Nomad.Matrix;
using Vortex.Network;
using Vortex.Activation.Kernels;
using Vortex.Cost.Kernels.Categorical;
using Vortex.Cost.Kernels.Legacy;
using Vortex.Initializer.Kernels;
using Vortex.Optimizer.Kernels;
using Vortex.Layer.Kernels;
using Vortex.Metric.Kernels.Categorical;
using Vortex.Pooler.Kernels;

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
            net.CreateLayer(new Dense(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));
            net.InitNetwork();
            net.InitNetwork();
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidOperationException), "Network is Locked.")]
        public void NetworkLockedExceptionLayer()
        {
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new Dense(3, new Tanh()));
            net.CreateLayer(new Output(1, new Tanh()));
            net.InitNetwork();
            net.CreateLayer(new Dense(3, new Tanh()));
        }

        [TestMethod]
        public void SequentialXor()
        {
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new Dense(3, new Tanh()));
            net.CreateLayer(new Dense(3, new Tanh(), null, new HeUniform()));
            net.CreateLayer(new Dense(3, new Tanh()));
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

            var accx = 0.0;
            for (var i = 0; i < 8; i++) accx = net.Train(inputs[i % 8], outputs[i % 8], new Accuracy());

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
            Trace.WriteLine(" Metrics Accuracy: " + accx);
            Assert.IsTrue(acc > 80.0, "Network did not learn XOR");
        }

        [TestMethod]
        public void DenseMnist()
        {
            // Load Train Data
            var lines = File.ReadAllLines("..\\..\\..\\..\\..\\datasets\\mnist_train.csv").ToList();

            var mnistLables = new List<Matrix>();
            var mnistData = new List<Matrix>();

            for (var j = 1; j < lines.Count; j++)
            {
                var t = lines[j];
                var data = t.Split(',').ToList();
                mnistLables.Add(new Matrix(10, 1).Fill(0));
                mnistLables[j - 1][int.Parse(data[0]), 0] = 1.0;

                var mnist = new Matrix(784, 1);
                for (var i = 1; i < data.Count; i++)
                {
                    mnist[i - 1, 0] = double.Parse(data[i]);
                }

                mnistData.Add(mnist);
            }

            // Load Test Data
            var testlines = File.ReadAllLines("..\\..\\..\\..\\..\\datasets\\mnist_test.csv").ToList();

            var mnistTestLables = new List<Matrix>();
            var mnistTestData = new List<Matrix>();

            for (var j = 1; j < testlines.Count; j++)
            {
                var t = testlines[j];
                var data = t.Split(',').ToList();
                mnistTestLables.Add(new Matrix(10, 1).Fill(0));
                mnistTestLables[j - 1][int.Parse(data[0]), 0] = 1.0;

                var mnist = new Matrix(784, 1);
                for (var i = 1; i < data.Count; i++)
                {
                    mnist[i - 1, 0] = double.Parse(data[i]);
                }

                mnistTestData.Add(mnist);
            }

            // Create Network
            var net = new Sequential(new CategoricalCrossEntropy(), new Adam(0.3), null, 128);
            net.CreateLayer(new Dense(784, new Identity()));
            net.CreateLayer(new Dense(128, new Tanh()));
            net.CreateLayer(new Output(10, new Softmax()));
            net.InitNetwork();

            // Train Network
            var acc = 0.0;
            for (var i = 0; i < mnistData.Count / 100; i++)
            {
                acc = net.Train(mnistData[i % mnistData.Count], mnistLables[i % mnistData.Count]);
            }

            // Test Network
            for (var i = 0; i < mnistTestData.Count / 100; i++)
            {
                var mat = net.Forward(mnistTestData[i % mnistTestData.Count]);
                var matx = mnistTestLables[i % mnistTestData.Count];
            }

            Trace.WriteLine(" Metrics Accuracy: " + acc);
            Assert.IsTrue(acc > 80.0, "Network did not learn MNIST");
        }

        [TestMethod]
        public void ConvolutionalMnist()
        {
            // Load Data
            var lines = File.ReadAllLines("..\\..\\..\\..\\..\\datasets\\mnist_train.csv").ToList();

            var mnistLables = new List<Matrix>();
            var mnistData = new List<Matrix>();

            for (var j = 1; j < lines.Count; j++)
            {
                var t = lines[j];
                var data = t.Split(',').ToList();
                mnistLables.Add(new Matrix(1, 1).Fill(int.Parse(data[0])));

                var mnist = new Matrix(784, 1);
                for (var i = 1; i < data.Count; i++)
                {
                    mnist[i - 1, 0] = double.Parse(data[i]);
                }

                mnistData.Add(mnist);
            }

            // Create Network
            var net = new Sequential(new QuadraticCost(), new NesterovMomentum(0.03));
            net.CreateLayer(new Convolutional(784, new Average(), new Tanh()));
            net.CreateLayer(new Convolutional(100, new Average(), new Tanh()));
            net.CreateLayer(new Dense(100, new Tanh()));
            net.CreateLayer(new Output(10, new Softmax()));
            net.InitNetwork();

            // Train Network
            var acc = 0.0;
            for (var i = 0; i < 800; i++)
            {
                acc = net.Train(mnistData[i % mnistData.Count], mnistLables[i % mnistData.Count], new Accuracy());
            }
        
            // Write Acc Result
            Trace.WriteLine(" Metrics Accuracy: " + acc);
            Assert.IsTrue(acc > 80.0, "Network did not learn MNIST");
        }
    }
}
