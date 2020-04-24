// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;

namespace Vortex.Cost
{
    /// <summary>
    /// "Exponential Cost"
    /// </summary>
    public class ExponentionalCost : Utility.BaseCost
    {
        public double Tao { get; set; }

        public ExponentionalCost(ExponentionalCostSettings settings) : base(settings)
        {
            Tao = settings.Tao;
        }

        public override double Forward(Matrix Actual, Matrix Expected)
        {
            double error = 0.0;
            
            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += Math.Pow((Actual[i, j] - Expected[i, j]), 2);
                }
            }

            error /= Tao;

            error = Math.Exp(error);

            error *= Tao;

            BatchCost += error;

            return error;
        }

        public override Matrix Backward(Matrix Actual, Matrix Expected)
        {
            double error = 0.0;

            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    error += Math.Pow((Actual[i, j] - Expected[i, j]), 2);
                }
            }

            error /= Tao;

            error = Math.Exp(error);

            error *= Tao;
            
            Matrix gradMatrix = Actual.Duplicate();
            for (int i = 0; i < Actual.Rows; i++)
            {
                for (int j = 0; j < Actual.Columns; j++)
                {
                    gradMatrix[i, j] = (Actual[i, j] - Expected[i, j]) * error;
                }
            }

            return gradMatrix;
        }

        public override ECostType Type() => ECostType.ExponentionalCost;
    }

    public class ExponentionalCostSettings : CostSettings
    {
        public double Tao { get; set; }

        public ExponentionalCostSettings(double tao) { Tao = tao; }

        public override ECostType Type() => ECostType.ExponentionalCost;
    }
}
