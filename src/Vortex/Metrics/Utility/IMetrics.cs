﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.


using Nomad.Matrix;

namespace Vortex.Metrics.Utility
{
    public interface IMetrics
    {
        public double Evaluate(Matrix actual, Matrix expected);
    }
}
