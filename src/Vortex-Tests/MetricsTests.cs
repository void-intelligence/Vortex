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
            var actual = new Matrix(4, 1);
            var expected = new Matrix(4, 1);
            actual.InRandomize();
            expected.InRandomize();

            var metric = new Accuracy();
            var e = metric.Evaluate(actual, expected);

            var val = 0.0;
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                val += Math.Abs(actual[i, j] - expected[i, j]) < 0.5 ? 0 : 1;

            val /= actual.Rows * actual.Columns;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, "Accuracy Metric Evaluate.");
        }

        [TestMethod]
        public void ArgMaxAccuracyTest()
        {
            var actual = new Matrix(4, 1);
            var expected = new Matrix(4, 1);
            actual.InRandomize();
            expected.InRandomize();

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

            Assert.IsTrue(Math.Abs(e - val) < 0.01, "ArgMax Accuracy Metric Evaluate.");
        }

        [TestMethod]
        public void F1ScoreTest()
        {
            var actual = new Matrix(4, 1);
            var expected = new Matrix(4, 1);
            actual.InRandomize();
            expected.InRandomize();

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

            Assert.IsTrue(Math.Abs(e - val) < 0.01, "F1Score Metric Evaluate.");
        }

        [TestMethod]
        public void PrecisionTest()
        {
            var actual = new Matrix(4, 1);
            var expected = new Matrix(4, 1);
            actual.InRandomize();
            expected.InRandomize();

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

            Assert.IsTrue(Math.Abs(e - val) < 0.01, "Precision Metric Evaluate.");
        }

        [TestMethod]
        public void RecallTest()
        {
            var actual = new Matrix(4, 1);
            var expected = new Matrix(4, 1);
            actual.InRandomize();
            expected.InRandomize();

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
            if (double.IsNaN(val)) val = 0.0;

            Assert.IsTrue(Math.Abs(e - val) < 0.01, "Recall Metric Evaluate.");
        }

    }
}
