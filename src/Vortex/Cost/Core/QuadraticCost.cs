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
    public class QuadraticCost : Utility.Cost
    {
        public QuadraticCost(QuadraticCostSettings settings) : base(settings) { }
        
        public override double Forward(Matrix Actual, Matrix Expected, int layerCount)
        {
            double error = 0.0;

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += Math.Pow((Actual[i, j] - Expected[i, j]), 2);
                }
            }

            error /= 2;

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix Actual, Matrix Expected, int layerCount)
        {
            return Actual - Expected;
        }

        public override string ToString()
        {
            return Type().ToString();
        }

        public override ECostType Type()
        {
            return ECostType.QuadraticCost;
        }
    }

    public class QuadraticCostSettings : CostSettings { }
}
