﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Categorical
{
    public class CategoricalCrossEntropy : BaseCost
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
                error += -expected[i, j] * Math.Log(actual[i, j] + double.Epsilon);

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            return (-1 * expected).HadamardDivision(actual + actual.Fill(double.Epsilon));
        }

        public override ECostType Type()
        {
            return ECostType.CategoricalCorssEntropy;
        }
    }
}