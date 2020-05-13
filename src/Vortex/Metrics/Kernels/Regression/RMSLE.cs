﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Metrics.Utility;
using Vortex.Cost.Kernels.Regression;

namespace Vortex.Metrics.Kernels.Regression
{
    public sealed class RMSLE : BaseMetrics
    {
        public RMSLE() : base(0.0)
        {
        }

        public override double Evaluate(Matrix actual, Matrix expected)
        {
            return Math.Sqrt(new MSLE().Evaluate(actual, expected));
        }

        public override EMetricsType Type()
        {
            return EMetricsType.RegressionRMSE;
        }
    }
}