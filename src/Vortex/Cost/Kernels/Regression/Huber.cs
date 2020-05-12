// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Regression
{
    public class Huber : BaseCost
    {
        public double Margin { get; set; }

        public Huber(double margin = 1.0)
        {
            Margin = margin;
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                var diff = Math.Abs(expected[i, j] - actual[i, j]);
                error += diff <= Margin ? 0.5 * diff * diff : Margin * (diff - 0.5 * Margin);
            }

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
            {
                var diff = Math.Abs(expected[i, j] - actual[i, j]);
                gradMatrix[i, j] = diff <= Margin
                    ? actual[i, j] - expected[i, j]
                    : Margin * Math.Sign((actual[i, j] - expected[i, j]));
            }

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.RegressionHuber;
        }
    }
}