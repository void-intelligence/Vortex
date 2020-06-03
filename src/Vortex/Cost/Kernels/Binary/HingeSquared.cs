// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Core;

namespace Vortex.Cost.Kernels.Binary
{
    /// <summary>
    /// "Exponential Cost"
    /// </summary>
    public class HingeSquared : BaseCost
    {
        public double Margin { get; set; }

        public HingeSquared(double margin = 1.0)
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
                error += actual[i, j] * expected[i, j] < Margin
                    ? Math.Pow(Margin - expected[i, j] * actual[i, j], 2)
                    : 1.0;

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                gradMatrix[i, j] = actual[i, j] * expected[i, j] < Margin
                    ? -2.0 * expected[i, j] * (Margin - expected[i, j] * actual[i, j])
                    : 0.0;

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.BinaryHingeSquared;
        }
    }
}