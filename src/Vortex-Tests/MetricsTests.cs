using System;// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Nomad.Matrix;
using Vortex.Metrics.Kernels;

namespace VortexTests
{
    [TestClass]
    public class Metrics
    {
        [TestMethod]
        public void AccuracyTest()
        {
            var actual = new Matrix(100, 1);
            var expected = new Matrix(100, 1);
            actual.InRandomize(0.25, 0.75);
            expected.InRandomize(0.25, 0.75);

            var metric = new Accuracy();
            var e = metric.Evaluate(actual, expected);

            var val = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                val += Math.Abs(actual[i, j] - expected[i, j]) < 0.5 ? 0 : 1;

            val /= actual.Rows * actual.Columns;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, metric.Type().ToString() + " Evaluate.");
        }

        [TestMethod]
        public void ArgMaxAccuracyTest()
        {
            var actual = new Matrix(100, 1);
            var expected = new Matrix(100, 1);
            actual.InRandomize(0.25,0.75);
            expected.InRandomize(0.25, 0.75);

            var metric = new ArgMaxAccuracy();
            var e = metric.Evaluate(actual, expected);

            var val = 0.0;
            var max = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                if (actual[i, j] > max)
                    max = actual[i, j];


            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                if (Math.Abs(actual[i, j] - max) < 0.5)
                    actual[i, j] = 1.0;


            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                val += Math.Abs(actual[i, j] - expected[i, j]) < 0.5 ? 0 : 1;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, metric.Type().ToString() + " Evaluate.");
        }

        [TestMethod]
        public void F1ScoreTest()
        {
            var actual = new Matrix(100, 1);
            var expected = new Matrix(100, 1);
            actual.InRandomize(0.25, 0.75);
            expected.InRandomize(0.25, 0.75);

            var metric = new F1Score();
            var e = metric.Evaluate(actual, expected);

            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (Math.Abs(actual[i, j]) < 0.5 && Math.Abs(expected[i, j]) < 0.5)
                {
                    val += 0.0;
                }
                else if (Math.Abs(actual[i, j] - 1.0) < 0.5 && Math.Abs(expected[i, j] - 1.0) < 0.5)
                {
                    val += 2.0;
                    div += 2.0;
                }

                div++;
            }

            val /= div;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, metric.Type().ToString() + " Evaluate.");
        }

        [TestMethod]
        public void PrecisionTest()
        {
            var actual = new Matrix(100, 1);
            var expected = new Matrix(100, 1);
            actual.InRandomize(0.25, 0.75);
            expected.InFill(1);

            var metric = new Precision();
            var e = metric.Evaluate(actual, expected);

            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (!(actual[i, j] >= 0.5)) continue;
                div++;
                if (Math.Abs(expected[i, j] - 1.0) < 0.1)
                    val++;
            }

            val /= div;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, metric.Type().ToString() + " Evaluate.");

            // Precision 0-1 test
            new Precision().Evaluate(actual.Fill(0), actual.Fill(1));

        }

        [TestMethod]
        public void RecallTest()
        {
            var actual = new Matrix(100, 1);
            var expected = new Matrix(100, 1);
            actual.InRandomize(0.5, 1.25);
            expected.InRandomize(0.5, 1.25);

            var metric = new Recall();
            var e = metric.Evaluate(actual, expected);

            var val = 0.0;
            var div = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                if (!(Math.Abs(expected[i, j] - 1.0) < 0.1)) continue;
                div++;
                if (actual[i, j] >= 0.5)
                    val++;
            }

            val /= div;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, metric.Type().ToString() + " Evaluate.");

            // Recall NaN test
            new Recall().Evaluate(actual.Fill(double.NaN), actual.Fill(double.NaN));
        }
    }
}
