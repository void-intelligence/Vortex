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
        public ItakuraSaitoDistance(ItakuraSaitoDistanceSettings settings) : base(settings) { }

        public override double Forward(Matrix Actual, Matrix Expected, int layerCount)
        {
            double error = 0.0;

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += (Expected[i, j] / Actual[i, j]) - Math.Log(Expected[i, j] - Actual[i, j]) - 1;
                }
            }

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix Actual, Matrix Expected, int layerCount)
        {
            Matrix gradMatrix = Actual.Duplicate();

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    gradMatrix[i, j] = (Actual[i, j] - Expected[i, j]) / Math.Pow(Actual[i, j], 2);
                }
            }

            return gradMatrix;
        }

        public override string ToString()
        {
            return Type().ToString();
        }

        public override ECostType Type()
        {
            return ECostType.ItakuraSaitoDistance;
        }
    }

    public class ItakuraSaitoDistanceSettings : CostSettings { }
}
