// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Core;

namespace Vortex.Cost.Kernels.Binary
{
    /// <summary>
    /// "Cross Entropy": Also known as "Bernoulli negative log-likelihood" and "Binary Cross-Entropy"
    /// </summary>
    public class CrossEntropy : BaseCost
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
                error += -expected[i, j] * Math.Log(actual[i, j]) - (1.0 - expected[i, j]) * Math.Log(1.0 - actual[i, j] + double.Epsilon);

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            return (expected - actual).HadamardDivision((actual.Hadamard(actual) - (actual + actual.Fill(double.Epsilon))));
        }

        public override ECostType Type()
        {
            return ECostType.BinaryCrossEntropy;
        }
    }
}
