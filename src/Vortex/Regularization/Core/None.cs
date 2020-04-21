// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization
{
    public sealed class None : Utility.Regularization
    {
        public None() : base(null) { }

        public override double CalculateNorm(Matrix X) => 0;

        public override string ToString() => Type().ToString();

        public override ERegularizationType Type() => ERegularizationType.None;
    }
}
