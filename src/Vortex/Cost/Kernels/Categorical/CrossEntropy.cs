// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Categorical
{
    public class CrossEntropy : BaseCost
    {
        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                error += -expected[i, j] * Math.Log(actual[i, j] + double.Epsilon);

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var oneover = (actual + actual.Fill(double.Epsilon)).OneOver();
            return (-1 * expected).Hadamard(oneover);
        }

        public override ECostType Type()
        {
            return ECostType.CategoricalCorssEntropy;
        }
    }
}