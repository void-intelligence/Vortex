// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation.Utility
{
    public interface IActivation
    {
        public Matrix Forward(Matrix input);

        public Matrix Backward(Matrix input);

        protected double Activate(double input);

        protected double Derivative(double input);

        public EActivationType Type();
    }
}