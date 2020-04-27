// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
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
    }
}
