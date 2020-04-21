// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost
{
    /// <summary>
    /// "Cross Entropy Cost": Also known as "Bernoulli negative log-likelihood" and "Binary Cross-Entropy"
    /// </summary>
    public class CrossEntropyCost : Utility.BaseCost
    {
        public CrossEntropyCost(CrossEntropyCostSettings settings) : base(settings) { }

        public override double Forward(Matrix Actual, Matrix Expected, int layerCount)
        {
            double error = 0.0;

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += -Expected[i, j] * Math.Log(Actual[i, j]) + (1.0 - Expected[i, j]) * Math.Log(1 - Actual[i, j]);
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
                    gradMatrix[i, j] = ((Actual[i, j] - Expected[i, j])) / ((1 - Actual[i, j]) * Actual[i, j]);
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
            return ECostType.CrossEntropyCost;
        }
    }

    public class CrossEntropyCostSettings : CostSettings { }
}
