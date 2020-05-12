// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Legacy
{
    /// <summary>
    /// "Hellinger Distance": needs to have positive values, and ideally values between 0 and 1. The same is true for the following divergences.
    /// </summary>
    public sealed class HellingerDistance : BaseCost
    {
        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += Math.Pow(Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j]), 2);

            error *= 1 / Math.Sqrt(2);
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            if (actual.Rows != expected.Rows || actual.Columns != expected.Columns) throw new ArgumentException("Actual Matrix does not have the same size as The Expected Matrix");

            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (Math.Sqrt(actual[i, j]) - Math.Sqrt(expected[i, j])) / (Math.Sqrt(2) * Math.Sqrt(actual[i, j]));

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.LegacyHellingerDistance;
        }
    }
}
