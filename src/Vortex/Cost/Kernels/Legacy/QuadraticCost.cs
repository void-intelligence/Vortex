// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Core;

namespace Vortex.Cost.Kernels.Legacy
{
    /// <summary>
    /// "Quadratic Cost": Also known as "Mean Squared Error" or "Maximum Likelihood" or "Sum Squared Error"
    /// </summary>
    public sealed class QuadraticCost : BaseCost
    {
        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += Math.Pow(actual[i, j] - expected[i, j], 2);

            error /= 2;
            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                gradMatrix[i, j] = Math.Pow(expected[i, j] - actual[i, j], 2);

            return gradMatrix * (1.0 / actual.Rows * actual.Columns);
        }

        public override ECostType Type()
        {
            return ECostType.LegacyQuadraticCost;
        }
    }
}
