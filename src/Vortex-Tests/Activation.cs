using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Text;
using Nomad.Matrix;
using Vortex.Activation;

namespace Vortex_Tests
{
    [TestClass]
    public class VortexActivation
    {
        [TestMethod]
        public void ArctanTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();

            // Activate
            a = new Arctan().Forward(a);

            // Test
            b.InMap(System.Math.Atan);

            Assert.IsTrue(a == b, "Arctan Activation successful");
        }

        [TestMethod]
        public void ArctanPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();

            // Activate
            a = new Arctan().Backward(a);

            // Test
            b.InMap((x) => 1 / (1 + System.Math.Pow(x, 2)));

            Assert.IsTrue(a == b, "Arctan Activation Derivative successful");
        }

        [TestMethod]
        public void BinaryStepTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();

            // Activate
            a = new BinaryStep().Forward(a);

            // Test
            b.InMap((x) => (x < 0) ? 0 : 1);

            Assert.IsTrue(a == b, "BinaryStep Activation successful");
        }

        [TestMethod]
        public void BinaryStepPrimeTest()
        {
            Matrix a = new Matrix(2, 2);
            a.InRandomize();
            Matrix b = a.Duplicate();

            // Activate
            a = new BinaryStep().Backward(a);

            // Test
            b.InMap((x) => 0);

            Assert.IsTrue(a == b, "BinaryStep Activation Derivative successful");
        }
    }
}
