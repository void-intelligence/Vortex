// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Cost.Kernels.Legacy;
using Vortex.Cost.Kernels.Binary;
using Vortex.Cost.Kernels.Categorical;
using Vortex.Cost.Kernels.Regression;

namespace VortexTests
{
    [TestClass]
    public class Cost
    {
        [TestClass]
        public class Binary
        {
            [TestMethod]
            public void CrossEntropyTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new CrossEntropy().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += -expected[i, j] * Math.Log(actual[i, j]) - (1.0 - expected[i, j]) * Math.Log(1.0 - actual[i, j] + double.Epsilon);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Cross Entropy Forward!");
                
                var autoDErr = new CrossEntropy().Backward(actual, expected);
                var oneover = (actual.Hadamard(actual) - (actual + actual.Fill(double.Epsilon))).OneOver();
                var dErr = (actual - expected).Hadamard(oneover);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Cross Entropy Backward!");
            }


            [TestMethod]
            public void ExponentialTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new Exponential().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Exp(-expected[i, j] * actual[i, j]);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Exponential Forward!");

                var autoDErr = new Exponential().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = -1 * Math.Exp(-expected[i, j] * actual[i, j]);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Exponential Backward!");
            }


            [TestMethod]
            public void HingeTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new Hinge().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Max(0.0, 1.0 - expected[i, j] * actual[i, j]);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Hinge Forward!");

                var autoDErr = new Hinge().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = -1 * Math.Exp(-expected[i, j] * actual[i, j]);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Hinge Backward!");
            }


            [TestMethod]
            public void HingeSquaredTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new HingeSquared().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += actual[i, j] * expected[i, j] < 1.0 ? Math.Pow(1.0 - expected[i, j] * actual[i, j], 2) : 1.0;
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Hinge Squared Forward!");


                var autoDErr = new HingeSquared().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = actual[i, j] * expected[i, j] < 1.0 ? -2.0 * expected[i, j] * (1.0 - expected[i, j] * actual[i, j]) : 0.0;
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Hinge Squared Backward!");
            }


            [TestMethod]
            public void LogitTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new Logit().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Log(1.0 + Math.Exp(actual[i, j] * -expected[i, j]));
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Logit Forward!");

                var autoDErr = new Logit().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = -1.0 / (1.0 + Math.Exp(actual[i, j] * expected[i, j]));
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Logit Backward!");
            }
        }

        [TestClass]
        public class Categorical
        {
            [TestMethod]
            public void CategoricalCrossEntropyTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new CategoricalCrossEntropy().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += -expected[i, j] * Math.Log(actual[i, j] + double.Epsilon);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Categorical Cross Entropy Forward!");

                var autoDErr = new CategoricalCrossEntropy().Backward(actual, expected);
                var oneover = (actual + actual.Fill(double.Epsilon)).OneOver();
                var dErr = (-1 * expected).Hadamard(oneover);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Categorical Cross Entropy Backward!");
            }

            [TestMethod]
            public void GeneralizedKLDTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new GeneralizedKLD().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]) - expected[i, j] + actual[i, j];
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Generalized KLD Forward!");

                var autoDErr = new GeneralizedKLD().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++) 
                    dErr[i, j] = (actual[i, j] - expected[i, j]) / actual[i, j];
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Generalized KLD Backward!");
            }

            [TestMethod]
            public void KLDTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new KLD().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += expected[i, j] * Math.Log(expected[i, j] / (actual[i, j] + double.Epsilon));
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "KLD Forward!");

                var autoDErr = new KLD().Backward(actual, expected);
                var oneover = (actual + actual.Fill(double.Epsilon)).OneOver();
                var dErr = (-1 * expected).Hadamard(oneover);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "KLD Backward!");
            }
        }

        [TestClass]
        public class Legacy
        {
            [TestMethod]
            public void HellingerDistanceTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new HellingerDistance().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Pow(Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j]), 2);
                error *= 1 / Math.Sqrt(2);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Hellinger Distance Forward!");

                var autoDErr = new HellingerDistance().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = (Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j])) / (Math.Sqrt(2) * Math.Sqrt(actual[i, j]));
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Hellinger Distance Backward!");
            }

            [TestMethod]
            public void SaitoDistanceTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new ItakuraSaitoDistance().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += expected[i, j] / (actual[i, j] + double.Epsilon) - Math.Log(expected[i, j] - actual[i, j]) - 1;
                if (double.IsNaN(error)) error = 0;
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Itakura Saito Distance Forward!");

                var autoDErr = new ItakuraSaitoDistance().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = (actual[i, j] - expected[i, j]) / Math.Pow(actual[i, j], 2);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Itakura Saito Distance Backward!");
            }

            [TestMethod]
            public void QuadraticTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new QuadraticCost().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Pow(actual[i, j] - expected[i, j], 2);
                error /= 2;
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Quadratic Forward!");

                var autoDErr = new QuadraticCost().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = Math.Pow(expected[i, j] - actual[i, j], 2);
                dErr *= 1.0 / actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Quadratic Backward!");
            }
        }

        [TestClass]
        public class Regression
        {
            [TestMethod]
            public void CosineProximityTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new CosineProximity().Forward(actual, expected);
                {
                    var dotProd = 0.0;
                    var dot = actual * expected.T();
                    for (var i = 0; i < dot.Rows; i++)
                    for (var j = 0; j < dot.Columns; j++)
                        dotProd += dot[i, j];

                    var aNorm = 0.0;
                    dot = actual * actual.T();
                    for (var i = 0; i < dot.Rows; i++)
                    for (var j = 0; j < dot.Columns; j++)
                        aNorm += dot[i, j];

                    var eNorm = 0.0;
                    dot = expected * expected.T();
                    for (var i = 0; i < dot.Rows; i++)
                    for (var j = 0; j < dot.Columns; j++)
                        eNorm = dot[i, j];
                    var error = dotProd / (aNorm * eNorm);
                    Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Cosine Proximity Forward!");
                }

                var autoDErr = new CosineProximity().Backward(actual, expected);
                {
                    var dotProd = 0.0;
                    var dot = actual * expected.T();
                    for (var i = 0; i < dot.Rows; i++)
                    for (var j = 0; j < dot.Columns; j++)
                        dotProd += dot[i, j];

                    var aNorm = 0.0;
                    dot = actual * actual.T();
                    for (var i = 0; i < dot.Rows; i++)
                    for (var j = 0; j < dot.Columns; j++)
                        aNorm += dot[i, j];

                    var eNorm = 0.0;
                    dot = expected * expected.T();
                    for (var i = 0; i < dot.Rows; i++)
                    for (var j = 0; j < dot.Columns; j++)
                        eNorm = dot[i, j];

                    var a = 1.0 / (eNorm * aNorm);
                    var b = dotProd / (aNorm * aNorm);

                    var dErr = new Matrix(actual.Rows, actual.Columns);
                    for (var i = 0; i < dErr.Rows; i++)
                    for (var j = 0; j < dErr.Columns; j++)
                        dErr[i, j] = a * (expected[i, j] - b * actual[i, j]);
                    Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Cosine Proximity Backward!");
                }
            }

            [TestMethod]
            public void HuberTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new Huber().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                {
                    var diff = Math.Abs(expected[i, j] - actual[i, j]);
                    error += diff <= 1.0 ? 0.5 * diff * diff : 1.0 * (diff - 0.5 * 1.0);
                }
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Huber Forward!");

                var autoDErr = new Huber().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                {
                    var diff = Math.Abs(expected[i, j] - actual[i, j]);
                    dErr[i, j] = diff <= 1.0
                        ? actual[i, j] - expected[i, j]
                        : 1.0 * Math.Sign(actual[i, j] - expected[i, j]);
                }
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Huber Backward!");
            }

            [TestMethod]
            public void LogCoshTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new LogCosh().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Log(Math.Cosh(expected[i, j] - actual[i, j]));
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Log Cosh Forward!");

                var autoDErr = new LogCosh().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = Math.Tanh(actual[i, j] - expected[i, j]);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Log Cosh Backward!");
            }

            [TestMethod]
            public void MAETest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new MAE().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Abs(expected[i, j] - actual[i, j]);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "MAE Forward!");

                var autoDErr = new MAE().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = Math.Sign(actual[i, j] - expected[i, j]);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "MAE Backward!");
            }

            [TestMethod]
            public void MAPETest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new MAPE().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Abs(expected[i, j] - actual[i, j]) / (expected[i, j] + double.Epsilon);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "MAPE Forward!");

                var autoDErr = new MAPE().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = actual[i, j] * (expected[i, j] - actual[i, j]) /
                    (Math.Pow(expected[i, j], 3.0) * Math.Abs(1.0 - actual[i, j] / (expected[i, j] + double.Epsilon)));
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "MAPE Backward!");
            }

            [TestMethod]
            public void MSETest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new MSE().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Pow(expected[i, j] - actual[i, j], 2);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "MSE Forward!");

                var autoDErr = new MSE().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = 2 * (actual[i, j] - expected[i, j]);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "MSE Backward!");
            }

            [TestMethod]
            public void MSLETest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new MSLE().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += Math.Pow(Math.Log((expected[i, j] + 1.0) / (actual[i, j] + 1.0)), 2.0);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "MSLE Forward!");

                var autoDErr = new MSLE().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = - 2.0 * Math.Log((expected[i, j] + 1.0) / (actual[i, j] + 1.0)) / (actual[i, j] + 1.0);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "MSLE Backward!");
            }

            [TestMethod]
            public void PoissonTest()
            {
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new Poisson().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    error += actual[i, j] - expected[i, j] * Math.Log(actual[i, j] + double.Epsilon);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Poisson Forward!");

                var autoDErr = new Poisson().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = 1.0 - expected[i, j] / (actual[i, j] + double.Epsilon);
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Poisson Backward!");
            }

            [TestMethod]
            public void QuantileTest()
            {
                const double tau = 2.0 * Math.PI;
                var actual = new Matrix(4, 1);
                var expected = new Matrix(4, 1);
                actual.InRandomize();
                expected.InRandomize();

                var autoErr = new Quantile().Forward(actual, expected);
                var error = 0.0;
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                        error += actual[i, j] - expected[i, j] >= 0.0
                            ? (tau - 1.0) * (expected[i, j] - actual[i, j])
                            : tau * (expected[i, j] - actual[i, j]);
                error /= actual.Rows * actual.Columns;
                Assert.IsTrue(Math.Abs(error - autoErr) < 0.01, "Quantile Forward!");

                var autoDErr = new Quantile().Backward(actual, expected);
                var dErr = new Matrix(actual.Rows, actual.Columns);
                for (var i = 0; i < actual.Rows; i++)
                for (var j = 0; j < actual.Columns; j++)
                    dErr[i, j] = actual[i, j] - expected[i, j] >= 0.0
                        ? 1.0 - tau
                        : -tau;
                Assert.IsTrue(Math.Abs(dErr.FrobeniusNorm() - autoDErr.FrobeniusNorm()) < 0.01, "Quantile Backward!");
            }
        }
    }
}