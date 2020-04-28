// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Cost;

namespace VortexTests
{
    [TestClass]
    public class VortexCost
    {
        [TestMethod]
        public void CrossEntropyTest()
        {
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += -expected[i, j] * Math.Log(actual[i, j]) + (1.0 - expected[i, j]) * Math.Log(1 - actual[i, j]);
                }
            }

            CrossEntropyCost cost = new CrossEntropyCost();
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Cross Entropy Cost successful");
        }

        [TestMethod]
        public void CrossEntropyPrimeTest()
        {
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = ((actual[i, j] - expected[i, j])) / ((1 - actual[i, j]) * actual[i, j]);
                }
            }

            CrossEntropyCost cost = new CrossEntropyCost();
            Matrix calculatedMatrix = cost.Backward(actual, expected);
            
            Assert.IsTrue(gradMatrix == calculatedMatrix, "Cross Entropy Cost Derivative successful");
        }

        [TestMethod]
        public void ExponentialCostTest()
        {
            Random rnd = new Random();
            double tao = rnd.NextDouble() * 10;
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((actual[i, j] - expected[i, j]), 2);
                }
            }

            error /= tao;
            error = Math.Exp(error);
            error *= tao;

            ExponentialCost cost = new ExponentialCost(new ExponentionalCostSettings(tao));
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Exponential Cost successful");
        }

        [TestMethod]
        public void ExponentialCostPrimeTest()
        {
            Random rnd = new Random();
            double tao = rnd.NextDouble() * 10;
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((actual[i, j] - expected[i, j]), 2);
                }
            }

            error /= tao;
            error = Math.Exp(error);
            error *= tao;

            Matrix gradMatrix = actual.Duplicate();
            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = (actual[i, j] - expected[i, j]) * error;
                }
            }

            ExponentialCost cost = new ExponentialCost(new ExponentionalCostSettings(tao));
            Matrix calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Exponential Cost Derivative successful");
        }

        [TestMethod]
        public void GeneralizedKullbackLeiblerDivergenceTest()
        {
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();


            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]) - expected[i, j] + actual[i, j];
                }
            }

            GeneralizedKullbackLeiblerDivergence cost = new GeneralizedKullbackLeiblerDivergence();
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Generalized Kullback Leibler Divergence successful");
        }

        [TestMethod]
        public void GeneralizedKullbackLeiblerDivergencePrimeTest()
        {
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / actual[i, j];
                }
            }

            GeneralizedKullbackLeiblerDivergence cost = new GeneralizedKullbackLeiblerDivergence();
            Matrix calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Generalized Kullback Leibler Divergence Derivative successful");
        }

        [TestMethod]
        public void HellingerDistanceTest()
        {
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j])), 2);
                }
            }
            error *= (1 / Math.Sqrt(2));

            HellingerDistance cost = new HellingerDistance();
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Hellinger Distance successful");
        }

        [TestMethod]
        public void HellingerDistancePrimeTest()
        {
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = (Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j])) / (Math.Sqrt(2) * Math.Sqrt(actual[i, j]));
                }
            }

            HellingerDistance cost = new HellingerDistance();
            Matrix calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Hellinger Distance Derivative successful");
        }

        [TestMethod]
        public void ItakuraSaitoDistanceTest()
        {
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += (expected[i, j] / actual[i, j]) - Math.Log(expected[i, j] - actual[i, j]) - 1;
                }
            }
            if (double.IsNaN(error))
            {
                error = 0;
            }

            ItakuraSaitoDistance cost = new ItakuraSaitoDistance();
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Itakura Saito Distance successful");
        }

        [TestMethod]
        public void ItakuraSaitoDistancePrimeTest()
        {
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / Math.Pow(actual[i, j], 2);
                }
            }

            ItakuraSaitoDistance cost = new ItakuraSaitoDistance();
            Matrix calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Itakura Saito Distance Derivative successful");
        }

        [TestMethod]
        public void KullbackLeiblerDivergenceTest()
        {
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]);
                }
            }

            KullbackLeiblerDivergence cost = new KullbackLeiblerDivergence();
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Kullback Leibler Divergence Cost successful");
        }

        [TestMethod]
        public void KullbackLeiblerDivergencePrimeTest()
        {
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = -(expected[i, j] / actual[i, j]);
                }
            }

            KullbackLeiblerDivergence cost = new KullbackLeiblerDivergence();
            Matrix calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Kullback Leibler Divergence Derivative successful");
        }

        [TestMethod]
        public void QuadraticCostTest()
        {
            double error = 0.0;
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((actual[i, j] - expected[i, j]), 2);
                }
            }

            error /= 2;

            QuadraticCost cost = new QuadraticCost();
            double calculatedError = cost.Forward(actual, expected);

            Assert.IsTrue(Math.Abs(error - calculatedError) < 0.01f, "Quadratic Cost successful");
        }

        [TestMethod]
        public void QuadraticCostPrimeTest()
        {
            Matrix actual = new Matrix(4, 1);
            actual.InRandomize();
            Matrix expected = new Matrix(4, 1);
            expected.InRandomize();

            Matrix gradMatrix = actual.Duplicate();
            gradMatrix = actual - expected;

            QuadraticCost cost = new QuadraticCost();
            Matrix calculatedMatrix = cost.Backward(actual, expected);

            Assert.IsTrue(gradMatrix == calculatedMatrix, "Quadratic Cost Derivative successful");
        }

    }
}
