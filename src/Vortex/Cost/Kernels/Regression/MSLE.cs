// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Core;

namespace Vortex.Cost.Kernels.Regression
{
    public class MSLE : BaseCost
    {
        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                error += Math.Pow(Math.Log((expected[i, j] + 1.0) / (actual[i, j] + 1.0)), 2.0);

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                gradMatrix[i, j] = -2.0 * Math.Log((expected[i, j] + 1.0) / (actual[i, j] + 1.0)) / (actual[i, j] + 1.0);

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.RegressionMSLE;
        }
    }
}