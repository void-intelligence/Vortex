﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    /// <summary>
    /// Ridge Regularization
    /// </summary>
    public sealed class L2Kernel : BaseRegularizationKernel
    {
        public L2Kernel(L2 settings) : base(settings) { Lambda = settings.Lambda; }

        public override double CalculateNorm(Matrix input)
        {
            return input.EuclideanNorm() * Lambda;
        }


        public override ERegularizationType Type()
        {
            return ERegularizationType.L2;
        }
    }

    public sealed class L2 : BaseRegularization
    {
        public L2(double lambda)
        {
            Lambda = lambda;
        }

        public override ERegularizationType Type()
        {
            return ERegularizationType.L2;
        }

        public double Lambda { get; set; }
    }
}
