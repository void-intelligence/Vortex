using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Mutation.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Mutation
    {
        [TestMethod]
        public void DefaultMutationTest()
        {
            var original = new Matrix(4, 1);
            original.InRandomize();
            var mutated = original.Duplicate();

            var mutation = new DefaultMutation();
            for (var i = 0; i < 100; i++) mutated.InMap(mutation.Mutate);

            Assert.IsTrue(Math.Abs(original.FrobeniusNorm() - mutated.FrobeniusNorm()) > 0.01, mutation.Type().ToString() + " Mutation!");
        }

        [TestMethod]
        public void NoMutationTest()
        {
            var original = new Matrix(4, 1);
            original.InRandomize();
            var mutated = original.Duplicate();

            var mutation = new NoMutation();
            for (var i = 0; i < 100; i++) mutated.InMap(mutation.Mutate);

            Assert.IsTrue(Math.Abs(original.FrobeniusNorm() - mutated.FrobeniusNorm()) < 0.01, mutation.Type().ToString() + " Mutation!");
        }
    }
}