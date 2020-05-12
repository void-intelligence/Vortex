// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Legacy
{
    /// <summary>
    /// "Itakura Saito Distance" function
    /// </summary>
    public class ItakuraSaitoDistance : BaseCost
    {
        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += expected[i, j] / actual[i, j] - Math.Log(expected[i, j] - actual[i, j]) - 1;
            if (double.IsNaN(error)) error = 0;

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / Math.Pow(actual[i, j], 2);

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.LegacyItakuraSaitoDistance;
        }
    }
}
