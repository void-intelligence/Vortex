﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Regression
{
    public class Quantile : BaseCost
    {
        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public double Tau { get; set; }

        public Quantile(double tau = 2.0 * Math.PI)
        {
            Tau = tau;
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                error += actual[i, j] - expected[i, j] >= 0.0
                    ? (Tau - 1.0) * (expected[i, j] - actual[i, j])
                    : Tau * (expected[i, j] - actual[i, j]);

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                gradMatrix[i, j] = actual[i, j] - expected[i, j] >= 0.0
                    ? 1.0 - Tau
                    : -Tau;

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.RegressionQuantile;
        }
    }
}