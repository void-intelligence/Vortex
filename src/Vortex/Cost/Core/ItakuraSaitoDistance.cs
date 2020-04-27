// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost
{
    /// <summary>
    /// "Itakura Saito Distance" cost function
    /// </summary>
    public class ItakuraSaitoDistance : Utility.BaseCost
    {
        public ItakuraSaitoDistance(ItakuraSaitoDistanceSettings settings = null) : base(settings) { }

        public override double Forward(Matrix actual, Matrix expected)
        {
            double error = 0.0;

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += (expected[i, j] / actual[i, j]) - Math.Log(expected[i, j] - actual[i, j]) - 1;
                }
            }
            if (double.IsNaN(error))
            {
                error = 0;
            }

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            Matrix gradMatrix = actual.Duplicate();

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / Math.Pow(actual[i, j], 2);
                }
            }

            return gradMatrix;
        }

        public override ECostType Type() => ECostType.ItakuraSaitoDistance;
    }

    public class ItakuraSaitoDistanceSettings : CostSettings
    {
        public override ECostType Type() => ECostType.ItakuraSaitoDistance;
    }
}
