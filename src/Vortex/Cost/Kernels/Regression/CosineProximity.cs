// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels.Regression
{
    public class CosineProximity : BaseCost
    {
        public override double Forward(Matrix actual, Matrix expected)
        {
            var dotProd = 0.0;
            var dot = actual * expected.T();
            for (var i = 0; i < dot.Rows; i++)
            for (var j = 0; j < dot.Columns; j++)
                dotProd += dot[i, j];

            var aNorm = 0.0;
            dot = actual * actual.T();
            for (var i = 0; i < dot.Rows; i++)
            for (var j = 0; j < dot.Columns; j++)
                aNorm += dot[i, j];

            var eNorm = 0.0;
            dot = expected * expected.T();
            for (var i = 0; i < dot.Rows; i++)
            for (var j = 0; j < dot.Columns; j++)
                eNorm = dot[i, j];

            var error = dotProd / (aNorm * eNorm);
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var dotProd = 0.0;
            var dot = actual * expected.T();
            for (var i = 0; i < dot.Rows; i++)
            for (var j = 0; j < dot.Columns; j++)
                dotProd += dot[i, j];

            var aNorm = 0.0;
            dot = actual * actual.T();
            for (var i = 0; i < dot.Rows; i++)
            for (var j = 0; j < dot.Columns; j++)
                aNorm += dot[i, j];

            var eNorm = 0.0;
            dot = expected * expected.T();
            for (var i = 0; i < dot.Rows; i++)
            for (var j = 0; j < dot.Columns; j++)
                eNorm = dot[i, j];

            var a = 1.0 / (eNorm * aNorm);
            var b = dotProd / (aNorm * aNorm);

            var gradMatrix = new Matrix(actual.Rows, actual.Columns);
            for (var i = 0; i < gradMatrix.Rows; i++)
            for (var j = 0; j < gradMatrix.Columns; j++)
                gradMatrix[i, j] = a * (expected[i, j] - b * actual[i, j]);

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.RegressionCosineProximity;
        }
    }
}