﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Core;

namespace Vortex.Cost.Kernels.Binary
{
    /// <summary>
    /// "Exponential Cost"
    /// </summary>
    public class Hinge : BaseCost
    {
        public double Margin { get; set; }

        public Hinge(double margin = 1.0)
        {
            Margin = margin;
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Forward(actual, expected);
        }

        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++)
                error += Math.Max(0.0, Margin - expected[i, j] * actual[i, j]);

            error /= actual.Rows * actual.Columns;
            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = new Matrix(actual.Rows, actual.Columns);
            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = -1 * Math.Exp(-expected[i, j] * actual[i, j]);

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.BinaryHinge;
        }
    }
}