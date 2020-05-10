﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Cost.Utility;
using Nomad.Matrix;

namespace Vortex.Cost.Kernels
{
    /// <summary>
    /// "Generalized Kullback Leibler Divergence": also known as "Bregman divergence"
    /// </summary>
    public class GeneralizedKullbackLeiblerDivergence : BaseCost
    {
        public override double Forward(Matrix actual, Matrix expected)
        {
            var error = 0.0;

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) error += expected[i, j] * Math.Log(expected[i, j] / actual[i, j]) - expected[i, j] + actual[i, j];

            BatchCost += error;
            return error;
        }

        public override Matrix Backward(Matrix actual, Matrix expected)
        {
            var gradMatrix = actual.Duplicate();

            for (var i = 0; i < actual.Rows; i++)
            for (var j = 0; j < actual.Columns; j++) gradMatrix[i, j] = (actual[i, j] - expected[i, j]) / actual[i, j];

            return gradMatrix;
        }

        public override ECostType Type()
        {
            return ECostType.GeneralizedKullbackLeiblerDivergence;
        }
    }
}