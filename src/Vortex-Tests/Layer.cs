using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Text;
using Nomad.Matrix;
using Vortex.Activation;
using Vortex.Cost;
using Vortex.Layer;
using Vortex.Optimizer;
using Vortex.Regularization;

namespace VortexTests
{
    [TestClass]
    public class VortexLayer
    {
        [TestMethod]
        public void FullyConnectedForwardTest()
        {
            FullyConnected fc =
                new FullyConnected(
                    new FullyConnectedSettings(
                        5, new TanhSettings(), new L2Settings(1)),
                    new GradientDescent(new GradientDescentSettings(0.001)));

            // Input
            Matrix x = new Matrix(fc.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            fc.Params["W"] = new Matrix(fc.NeuronCount, fc.NeuronCount);
            fc.Params["W"].InRandomize();
            fc.Params["B"] = new Matrix(fc.NeuronCount, 1);
            fc.Params["B"].InRandomize();

            // Forward
            Matrix testA = fc.Forward(x);

            //// MANUAL
            // Forward
            fc.Params["Z"] = (fc.Params["W"].T() * x) + fc.Params["B"];
            fc.Params["A"] = fc.ActivationFunction.Forward(fc.Params["Z"]);
            Matrix a = fc.Params["A"];

            Assert.IsTrue(a == testA, "Fully connected Layer function Forward operation successful");
        }

        [TestMethod]
        public void FullyConnectedBackwardTest()
        {
            FullyConnected fc =
                new FullyConnected(
                    new FullyConnectedSettings(
                        5, new TanhSettings(), new L2Settings(1)),
                    new GradientDescent(new GradientDescentSettings(0.001)));

            // Input
            Matrix x = new Matrix(fc.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            fc.Params["W"] = new Matrix(fc.NeuronCount, fc.NeuronCount);
            fc.Params["W"].InRandomize();
            fc.Params["B"] = new Matrix(fc.NeuronCount, 1);
            fc.Params["B"].InRandomize();

            // Forward
            Matrix testA = fc.Forward(x);
            Matrix res = testA.Duplicate();
            res.InRandomize();

            // Cost
            QuadraticCost cost = new QuadraticCost();
            double error = cost.Forward(testA, res);

            // Backward
            Matrix da = cost.Backward(testA, res);
            Matrix prevDa = fc.Backward(da);

            //// MANUAL
            Matrix gprime = fc.ActivationFunction.Backward(fc.Params["Z"]);
            Matrix dz = da.Hadamard(gprime);
            Matrix dw = dz;
            Matrix db = dz;
            Matrix manualPrevDa = fc.Params["W"] * dz;

            Assert.IsTrue(prevDa == manualPrevDa, "DA calculation successful");
            Assert.IsTrue(dw == fc.Grads["DW"], "DW calculation successful");
            Assert.IsTrue(db == fc.Grads["DB"], "DB calculation successful");
        }

        [TestMethod]
        public void FullyConnectedOptimizeTest()
        {
            FullyConnected fc =
                new FullyConnected(
                    new FullyConnectedSettings(
                        5, new TanhSettings(), new L2Settings(1)),
                    new GradientDescent(new GradientDescentSettings(0.001)));
            // Input
            Matrix x = new Matrix(fc.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            fc.Params["W"] = new Matrix(fc.NeuronCount, fc.NeuronCount);
            fc.Params["W"].InRandomize();
            fc.Params["B"] = new Matrix(fc.NeuronCount, 1);
            fc.Params["B"].InRandomize();

            // Forward
            Matrix testA = fc.Forward(x);
            Matrix res = testA.Duplicate();
            res.InRandomize();

            // Cost
            QuadraticCost cost = new QuadraticCost();
            double error = cost.Forward(testA, res);

            // Backward
            Matrix da = cost.Backward(testA, res);
            Matrix prevDa = fc.Backward(da);
        }

        [TestMethod]
        public void DropoutForwardTest()
        {
            Dropout dpDropout =
                new Dropout(
                    new DropoutSettings(
                        5, new TanhSettings(), new L2Settings(1), 1.0f),
                    new GradientDescent(new GradientDescentSettings(0.001)));

            // Input
            Matrix x = new Matrix(dpDropout.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            dpDropout.Params["W"] = new Matrix(dpDropout.NeuronCount, dpDropout.NeuronCount);
            dpDropout.Params["W"].InRandomize();
            dpDropout.Params["B"] = new Matrix(dpDropout.NeuronCount, 1);
            dpDropout.Params["B"].InRandomize();

            // Forward
            Matrix testA = dpDropout.Forward(x);
            Matrix testB = Matrix.Zero(testA.Rows, testA.Columns);

            Assert.IsTrue(testA == testB, "Dropout Layer function Forward operation successful");
        }

        [TestMethod]
        public void DropoutBackwardTest()
        {
            Dropout dpDropout =
                new Dropout(
                    new DropoutSettings(
                        5, new TanhSettings(), new L2Settings(1), 1.0f),
                    new GradientDescent(new GradientDescentSettings(0.001)));

            // Input
            Matrix x = new Matrix(dpDropout.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            dpDropout.Params["W"] = new Matrix(dpDropout.NeuronCount, dpDropout.NeuronCount);
            dpDropout.Params["W"].InRandomize();
            dpDropout.Params["B"] = new Matrix(dpDropout.NeuronCount, 1);
            dpDropout.Params["B"].InRandomize();

            // Forward
            Matrix testA = dpDropout.Forward(x);
            Matrix testB = testA.Duplicate();

            // Cost
            QuadraticCost cost = new QuadraticCost();
            double error = cost.Forward(testA, testB);

            // Backward
            Matrix da = cost.Backward(testA, testB);
            Matrix prevDa = dpDropout.Backward(da);
            Matrix testDa = Matrix.Zero(prevDa.Rows, prevDa.Columns);

            Assert.IsFalse(prevDa != testDa, "Dropout Layer function Backward operation successful");
        }

        [TestMethod]
        public void OutputForwardTest()
        {
            Output fc =
                new Output(
                    new OutputSettings(
                        5, new TanhSettings(), new L2Settings(1)),
                    new GradientDescent(new GradientDescentSettings(0.001)));

            // Input
            Matrix x = new Matrix(fc.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            fc.Params["W"] = new Matrix(fc.NeuronCount, fc.NeuronCount);
            fc.Params["W"].InRandomize();
            fc.Params["B"] = new Matrix(fc.NeuronCount, 1);
            fc.Params["B"].InRandomize();

            // Forward
            Matrix testA = fc.Forward(x);

            //// MANUAL
            // Forward
            fc.Params["Z"] = (fc.Params["W"].T() * x) + fc.Params["B"];
            fc.Params["A"] = fc.ActivationFunction.Forward(fc.Params["Z"]);
            Matrix a = fc.Params["A"];

            Assert.IsTrue(a == testA, "Fully connected Layer function Forward operation successful");
        }

        [TestMethod]
        public void OutputBackwardTest()
        {
            Output fc =
                new Output(
                    new OutputSettings(
                        5, new TanhSettings(), new L2Settings(1)),
                    new GradientDescent(new GradientDescentSettings(0.001)));

            // Input
            Matrix x = new Matrix(fc.NeuronCount, 1);
            x.InRandomize();

            //// LAYER
            // Init W and B
            fc.Params["W"] = new Matrix(fc.NeuronCount, fc.NeuronCount);
            fc.Params["W"].InRandomize();
            fc.Params["B"] = new Matrix(fc.NeuronCount, 1);
            fc.Params["B"].InRandomize();

            // Forward
            Matrix testA = fc.Forward(x);
            Matrix res = testA.Duplicate();
            res.InRandomize();

            // Cost
            QuadraticCost cost = new QuadraticCost();
            double error = cost.Forward(testA, res);

            // Backward
            Matrix da = cost.Backward(testA, res);
            Matrix prevDa = fc.Backward(da);

            //// MANUAL

            Matrix gprime = fc.ActivationFunction.Backward(fc.Params["Z"]);
            Matrix dz = da.Hadamard(gprime);
            Matrix dw = dz;
            Matrix db = dz;
            Matrix manualPrevDa = fc.Params["W"].T() * dz;

            Assert.IsTrue(prevDa == manualPrevDa, "DA calculation successful");
            Assert.IsTrue(dw == fc.Grads["DW"], "DW calculation successful");
            Assert.IsTrue(db == fc.Grads["DB"], "DB calculation successful");
        }

    }
}
