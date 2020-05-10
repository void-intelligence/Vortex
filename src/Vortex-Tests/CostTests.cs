// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Cost.Kernels.Legacy;

namespace VortexTests
{
    [TestClass]
    public class VortexCost
    {
        [TestMethod]
        public void GeneralizedKullbackLeiblerDivergenceTest()
        {
            var error = 0.0;
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();


            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]) - expected[i, j] + actual[i, j];

            var cost = new GeneralizedKullbackLeiblerDivergence();
            var calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Generalized Kullback Leibler Divergence successful");
        }

        [TestMethod]
        public void GeneralizedKullbackLeiblerDivergencePrimeTest()
        {
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / actual[i, j];

            var cost = new GeneralizedKullbackLeiblerDivergence();
            var calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Generalized Kullback Leibler Divergence Derivative successful");
        }

        [TestMethod]
        public void HellingerDistanceTest()
        {
            var error = 0.0;
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += Math.Pow(Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j]), 2);
            error *= 1 / Math.Sqrt(2);

            var cost = new HellingerDistance();
            var calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Hellinger Distance successful");
        }

        [TestMethod]
        public void HellingerDistancePrimeTest()
        {
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j])) / (Math.Sqrt(2) * Math.Sqrt(actual[i, j]));

            var cost = new HellingerDistance();
            var calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Hellinger Distance Derivative successful");
        }

        [TestMethod]
        public void ItakuraSaitoDistanceTest()
        {
            var error = 0.0;
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += expected[i, j] / actual[i, j] - Math.Log(expected[i, j] - actual[i, j]) - 1;
            if (double.IsNaN(error)) error = 0;

            var cost = new ItakuraSaitoDistance();
            var calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Itakura Saito Distance successful");
        }

        [TestMethod]
        public void ItakuraSaitoDistancePrimeTest()
        {
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / Math.Pow(actual[i, j], 2);

            var cost = new ItakuraSaitoDistance();
            var calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Itakura Saito Distance Derivative successful");
        }

        [TestMethod]
        public void KullbackLeiblerDivergenceTest()
        {
            var error = 0.0;
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]);

            var cost = new KullbackLeiblerDivergence();
            var calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Kullback Leibler Divergence Cost successful");
        }

        [TestMethod]
        public void KullbackLeiblerDivergencePrimeTest()
        {
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = -(expected[i, j] / actual[i, j]);

            var cost = new KullbackLeiblerDivergence();
            var calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Kullback Leibler Divergence Derivative successful");
        }

        [TestMethod]
        public void QuadraticCostTest()
        {
            var error = 0.0;
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += Math.Pow(actual[i, j] - expected[i, j], 2);

            error /= 2;

            var cost = new QuadraticCost();
            var calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Quadratic Cost successful");
        }

        [TestMethod]
        public void QuadraticCostPrimeTest()
        {
            var actual = new Matrix(4, 1);
            actual.InRandomize();
            var expected = new Matrix(4, 1);
            expected.InRandomize();

            var n = 0;
            var da = actual.Duplicate();
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                n++;
                da[i, j] = Math.Pow(expected[i, j] - actual[i, j], 2);
            }

            var gradMatrix = da * (1.0 / n);


            var cost = new QuadraticCost();
            var calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Quadratic Cost Derivative successful");
        }

    }
}
