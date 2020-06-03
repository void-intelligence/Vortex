// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Core;
using Vortex.Initializer.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Initializer
    {
        [TestMethod]
        public void AutoTest()
        {
            var init = new Auto();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void ConstTest()
        {
            var init = new Const();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void GlorotNormalTest()
        {
            var init = new GlorotNormal();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void GlorotUniformTest()
        {
            var init = new GlorotUniform();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void HeNormalTest()
        {
            var init = new HeNormal();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }


        [TestMethod]
        public void HeUniformTest()
        {
            var init = new HeUniform();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }


        [TestMethod]
        public void LeCunNormalTest()
        {
            var init = new LeCunNormal();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void LeCunUniformTest()
        {
            var init = new LeCunUniform();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void NormalTest()
        {
            var init = new Normal();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void OneTest()
        {
            var init = new One();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void UniformTest()
        {
            var init = new Uniform();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) > 0.01, init.Type().ToString() + " Initializer.");
        }

        [TestMethod]
        public void ZeroTest()
        {
            var init = new Zero();
            var m = init.Initialize(new Matrix(2, 2));
            Assert.IsTrue(Math.Abs(m.FrobeniusNorm()) < 0.01, init.Type().ToString() + " Initializer.");
        }
    }
}
