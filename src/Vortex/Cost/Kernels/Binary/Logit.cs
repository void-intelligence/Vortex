// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Binary
{
    /// <summary>
    /// "Exponential Cost"
    /// </summary>
    public class Logit : BaseCost
    {
        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                error += Math.Log(1.0 + Math.Exp(actual[i, j] * -expected[i, j]));

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                gradMatrix[i, j] = -1.0 / (1.0 + Math.Exp(actual[i, j] * expected[i, j]));

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.BinaryLogit;
        }
    }
}