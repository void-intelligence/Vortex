// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost
{
    /// <summary>
    /// "Quadratic Cost": Also known as "Mean Squared Error" or "Maximum Likelihood" or "Sum Squared Error"
    /// </summary>
    public class QuadraticCost : Utility.BaseCost
    {
        public QuadraticCost(QuadraticCostSettings settings = null) : base(settings) { }
        
        public override double Forward(Matrix actual, Matrix expected)
        {
            double error = 0.0;

            for (int i = 0; i < actual.Rows; i++)
            {
                for (int j = 0; j < actual.Columns; j++)
                {
                    error += Math.Pow((actual[i, j] - expected[i, j]), 2);
                }
            }

            error /= 2;

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            return actual - expected;
        }

        public override ECostType Type() => ECostType.QuadraticCost;
    }

    public class QuadraticCostSettings : CostSettings
    {
        public override ECostType Type() => ECostType.QuadraticCost;
    }
}
